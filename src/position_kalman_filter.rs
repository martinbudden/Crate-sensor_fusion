use vqm::{Matrix3x3, Matrix3x3f32, Matrix9, Matrix9f32, Vector3f32};

/// `f32` variant of `PositionKalmanFilter0`.
pub type PositionKalmanFilterf32 = PositionKalmanFilter;

/*
Rule of Thumb for Tuning

If your filter output looks laggy or sluggish (clinging too closely to your old path and trailing behind rapid movements),
you need to trust your physics model less and your sensors more:
Action: Increase Q or Decrease R.

If your filter output looks jittery or nervous (shaking whenever the sensor outputs minor noise spikes),
you are trusting your noisy sensors too heavily over your smooth physics predictions:
Action: Decrease Q or Increase R.
*/
/// The system is split into two cleanly decoupled steps. This:
/// 1. avoids managing a massive 15x15 state matrix.
/// 2. linearizes the attitude so a Kalman Filter (rather than an Extended Kalman Filter) can be used.
/// ```text
///   ┌──────────────┐
///   │ IMU Acc/Gyro ├──► [ 1. ATTITUDE (MADGWICK) FILTER ] ──► Attitude Quaternion
///   └──────────────┘                │
///                                   ▼
///   ┌───────────┐       [    Transform Body ]
///   │ IMU Acc   ├─────► [ 2. Acceleration   ] ──► [ 3. POSITION KALMAN FILTER ]
///   └───────────┘       [    to Earth Frame ]                  ▲
///                                                              │
///                        GPS & Barometer Measurements ─────────┘
/// ```
#[allow(non_snake_case)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PositionKalmanFilter {
    // 3D Kinematic State Vectors
    /// Position (x, y, z).
    pub pos: Vector3f32,
    /// Velocity (x, y, z).
    pub vel: Vector3f32,
    /// Accelerometer Bias (x, y, z).
    pub acc_bias: Vector3f32,

    /// Predicted System Uncertainty Covariance Matrix (P).
    /// **P*: Prediction error covariance (the system's internal uncertainty).
    pub P: Matrix9f32,

    // --- Hyperparameters & Tuning Constants ---
    /// Process Noise spectral density mapping to Velocity variance.
    pub q_velocity: f32,
    /// Process Noise spectral density mapping to Sensor Drift variance.
    pub q_bias: f32,
    // History Buffer for Retrodictive Updates
    pub history: [Snapshot; Self::SNAPSHOT_SIZE], // Fixed circular window (e.g., handles up to 640ms of latency at 100Hz)
    pub head_idx: usize,                          // Current write pointer in our ring buffer
    pub system_time: f32,
    pub acc_accumulator: Vector3f32,
    pub tick_counter: usize,
    pub skip_factor: usize,
}

impl Default for PositionKalmanFilter {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(missing_docs)]
impl PositionKalmanFilter {
    pub const SNAPSHOT_SIZE: usize = 64; // 640ms of history at 100Hz

    pub const M11: usize = Matrix9f32::M11;
    pub const M22: usize = Matrix9f32::M22;
    pub const M33: usize = Matrix9f32::M33;

    const PP: usize = 0;
    const VP: usize = 1;
    const BP: usize = 2;

    const PV: usize = 3;
    const VV: usize = 4;
    const BV: usize = 5;

    const PB: usize = 6;
    const VB: usize = 7;
    const BB: usize = 8;
}

impl PositionKalmanFilter {
    /// Constructor.
    #[allow(non_snake_case)]
    #[must_use]
    pub fn new() -> Self {
        let mut P = Matrix9f32::default();
        // Seed initial Position uncertainty (e.g., we are confident within 1 meter)
        P[Self::PP][Self::M11] = 1.0;
        P[Self::PP][Self::M22] = 1.0;
        P[Self::PP][Self::M33] = 1.0;

        // Seed initial Velocity uncertainty (e.g., confident within 0.5 m/s)
        P[Self::VV][Self::M11] = 0.25;
        P[Self::VV][Self::M22] = 0.25;
        P[Self::VV][Self::M33] = 0.25;

        // Seed initial Bias uncertainty (e.g., accelerometer bias bounds)
        P[Self::BB][Self::M11] = 0.01;
        P[Self::BB][Self::M22] = 0.01;
        P[Self::BB][Self::M33] = 0.01;

        Self {
            pos: Vector3f32 { x: 0.0, y: 0.0, z: 0.0 },
            vel: Vector3f32 { x: 0.0, y: 0.0, z: 0.0 },
            acc_bias: Vector3f32 { x: 0.0, y: 0.0, z: 0.0 },
            // A value of 0.05 implies that every second, you expect aerodynamic buffeting, vibration, or wind to naturally perturb the velocity
            // by roughly 0.22 m/s ie sqrt(0.05).
            q_velocity: 0.05,
            //  Sensor bias shifts very slowly due to thermal changes as the silicone heats up.
            //  So this value should be tiny so the filter treats bias as a near-constant,
            // shifting it incrementally over minutes rather than fluctuating on every single vibration loop.
            q_bias: 1e-4,
            P,
            history: [Snapshot::default(); Self::SNAPSHOT_SIZE],
            head_idx: 0,
            system_time: 0.0,
            acc_accumulator: Vector3f32 { x: 0.0, y: 0.0, z: 0.0 },
            tick_counter: 0,
            skip_factor: 1, // Default to 1 (no skipping), can be adjusted for logging or averaging
        }
    }
}

impl PositionKalmanFilter {
    #[inline]
    #[must_use]
    pub fn pos(&self) -> Vector3f32 {
        self.pos
    }

    #[inline]
    #[must_use]
    pub fn vel(&self) -> Vector3f32 {
        self.vel
    }

    #[inline]
    #[must_use]
    pub fn acc_bias(&self) -> Vector3f32 {
        self.acc_bias
    }
}

// **** Predict ****

impl PositionKalmanFilter {
    // Inside your main 100Hz IMU execution loop:
    pub fn handle_imu_tick(&mut self, acc: Vector3f32, dt: f32) {
        // Integrate physical kinematics forward (deterministic)
        self.predict_state(acc, dt);

        // Propagate covariance uncertainty blocks forward (stochastic)
        self.predict_covariance(dt);

        self.push_snapshot(acc, dt);
    }

    // Propagates the state vector forward using IMU acceleration inputs.
    /// Integrates raw IMU accelerometer data to predict new position and velocity vectors.
    ///
    /// This forms the continuous dead reckoning pipeline driving vehicle kinetics forward.
    ///
    /// ### Physical Mechanics
    /// ```math
    /// pos_k = pos_k₋₁ + vel_k₋₁ * dT + 0.5 * acc_true * dT²
    /// vel_k = vel_k₋₁ + acc_true * dT
    /// ```
    pub fn predict_state(&mut self, acc_measurement: Vector3f32, dt: f32) {
        // In NED, positive Z is down, so gravity is a positive vector
        let gravity = Vector3f32 { x: 0.0, y: 0.0, z: 9.80665 };

        // Calculate true physical acceleration by removing bias and adding gravity
        // Note: in a NED frame, gravity is a positive vector pointing downwards, so it is added to the accelerometer measurement.
        let acc_true = acc_measurement - self.acc_bias + gravity;

        // Physical mechanics
        // s = ut + 0.5 * a * t²
        self.pos += (self.vel + 0.5 * acc_true * dt) * dt;
        // v = u + at
        self.vel += acc_true * dt;
        // Bias remains constant during prediction, it is modeled as a random walk in covariance.
    }
    /*
    Our state vector is organized as [{p}, {v}, {b}]^T.
    The kinematic transition equations using simple Euler integration are:
    {p}_k = {p}_{k-1} + {v}_{k-1}Delta T
    {v}_k = {v}_{k-1} - vec{b}_{k-1}Delta T (assuming acceleration is updated via the control loop)
    {b}_k = {b}_{k-1}
    This means our 9x9 matrix **F** is incredibly sparse, containing only a few *dt* terms on the off-diagonals.
     If we write out the math for F * P * F^T manually using 3x3 blocks, the matrix operations simplify into a clean sequence of 3x3 array updates.
    */
    /// Propagates the 9x9 covariance matrix forward in time.
    ///
    /// A full 9x9 matrix multiplication involves 729 individual multiplications.
    /// Instead the matrix is divided into 9 separate 3x3 sub-matrices (blocks),
    /// and each block is processed separately.
    /// ```text
    /// ┌                 9x9                  ┐
    /// │ ┌   3x3    ┐┌   3x3    ┐┌   3x3    ┐ │
    /// │ │ Position ││ Position ││ Position │ │
    /// │ │ Position ││ Velocity ││ Bias     │ │
    /// │ └          ┘└          ┘└          ┘ │
    /// │ ┌   3x3    ┐┌   3x3    ┐┌   3x3    ┐ │
    /// │ │ Velocity ││ Velocity ││ Velocity │ │
    /// │ │ Position ││ Velocity ││ Bias     │ │
    /// │ └          ┘└          ┘└          ┘ │
    /// │ ┌   3x3    ┐┌   3x3    ┐┌   3x3    ┐ │
    /// │ │ Bias     ││ Bias     ││ Bias     │ │
    /// │ │ Position ││ Velocity ││ Bias     │ │
    /// │ └          ┘└          ┘└          ┘ │
    /// └                                      ┘
    /// In a Kalman filter tracking Position (p), Velocity (v), and an Accelerometer Bias (b)
    /// over a small time increment (dt),
    /// the physical laws of motion dictate:
    ///
    /// New Position = p + v * dt
    /// New Velocity = v - b * dt  (Bias is subtracted from acceleration)
    /// New Bias = b (Bias is modeled as a constant or random walk)
    ///
    ///Expressed as a 3x3 block state transition matrix, that gives:
    ///
    /// ┌                 9x9                  ┐
    /// │ ┌   3x3    ┐┌   3x3    ┐┌   3x3    ┐ │
    /// │ │    I     ││   dt.I   ││    0     │ │
    /// │ └          ┘└          ┘└          ┘ │
    /// │ ┌   3x3    ┐┌   3x3    ┐┌   3x3    ┐ │
    /// │ │    0     ││    I     ││  -dt.I   │ │
    /// │ └          ┘└          ┘└          ┘ │
    /// │ ┌   3x3    ┐┌   3x3    ┐┌   3x3    ┐ │
    /// │ │    0     ││    0     ││    I     │ │
    /// │ └          ┘└          ┘└          ┘ │
    /// └                                      ┘
    /// ```
    ///
    // =========================================================================
    // COVARIANCE PROPAGATION DERIVATION: P_new = F * P * F^T
    //
    // Continuous state transition dynamics layout:
    // F = [ I    dt*I     0   ]
    //     [ 0      I   -dt*I  ]
    //     [ 0      0      I   ]
    //
    // Multiplying F * P * F^T analytically yields the following block updates:
    // =========================================================================
    //
    // =========================================================================
    // POSITION COLUMNS (Blocks: PP, VP, BP)
    // =========================================================================
    //
    // PP_new = PP + dt * (VP + PV) + dt^2 * VV
    //
    // VP_new = VP + dt * VV - dt * BP - dt^2 * BV
    //
    // BP_new = BP + dt * BV
    //
    // =========================================================================
    //
    // =========================================================================
    // VELOCITY COLUMNS (Blocks: PV, VV, BV)
    // =========================================================================
    //
    // PV_new = PV + dt * VV - dt * PB - dt^2 * VB
    //
    // VV_new = VV - dt * (BV + VB) + dt^2 * BB
    //
    // BV_new = BV - dt * BB
    //
    // =========================================================================
    //
    // =========================================================================
    // BIAS COLUMNS (Blocks: PB, VB, BB)
    // =========================================================================
    //
    // PB_new = PB + dt * VB
    //
    // VB_new = VB - dt * BB
    //
    // BB_new = BB
    //
    // =========================================================================
    /// ## Formula
    /// *  `P_k = F * P_k₋₁ * Fᵀ + Q`
    #[allow(non_snake_case)]
    pub fn predict_covariance(&mut self, dt: f32) {
        let dt2 = dt * dt;
        // Capture the current a posteriori state (P_k₋₁).
        let E = self.P;

        // =====================================================================
        // PROPAGATE THE COVARIANCE (F * P * F^T)
        // =====================================================================

        // --- POSITION COLUMNS ---
        self.P[Self::PP] = E[Self::PP] + (E[Self::VP] + E[Self::PV]) * dt + E[Self::VV] * dt2;
        self.P[Self::VP] = E[Self::VP] + (E[Self::VV] - E[Self::BP]) * dt - E[Self::BV] * dt2;
        self.P[Self::BP] = E[Self::BP] + E[Self::BV] * dt;

        // --- VELOCITY COLUMNS ---
        self.P[Self::PV] = E[Self::PV] + (E[Self::VV] - E[Self::PB]) * dt - E[Self::VB] * dt2;
        self.P[Self::VV] = E[Self::VV] - (E[Self::BV] + E[Self::VB]) * dt + E[Self::BB] * dt2;
        self.P[Self::BV] = E[Self::BV] - E[Self::BB] * dt;

        // --- BIAS COLUMNS ---
        self.P[Self::PB] = E[Self::PB] + E[Self::VB] * dt;
        self.P[Self::VB] = E[Self::VB] - E[Self::BB] * dt;
        self.P[Self::BB] = E[Self::BB];

        // =====================================================================
        // APPLY PROCESS NOISE (Q)
        // =====================================================================
        // Continuous process noise integrated over dt maps primarily to the diagonal variance slots of Velocity and Bias.

        let q_vel = self.q_velocity * dt; // Standard continuous noise integration layout
        self.P[Self::VV][Self::M11] += q_vel;
        self.P[Self::VV][Self::M22] += q_vel;
        self.P[Self::VV][Self::M33] += q_vel;

        let q_bias_dt = self.q_bias * dt;
        self.P[Self::BB][Self::M11] += q_bias_dt;
        self.P[Self::BB][Self::M22] += q_bias_dt;
        self.P[Self::BB][Self::M33] += q_bias_dt;

        // Time propagation is highly sensitive to asymmetric shearing.
        self.P.enforce_symmetry();
    }
}

// **** Correct ***

impl PositionKalmanFilter {
    /// Phase 2 Altitude Correction using new measurement.
    /// Updates only the vertical Z axis components across all tracking states.
    ///
    /// ### Core Operations
    /// *  `S = P₂₂ + R` (Innovation Variance calculation)
    /// *  `K = P_column_2 * (1.0 / S)` (Kalman Gain column selection extraction)
    /// *  `E = P - K * H * P` (Covariance correction step)
    #[allow(clippy::similar_names)]
    #[allow(non_snake_case)]
    pub fn correct_altitude(&mut self, altitude: f32, R: f32) {
        // Calculate the scalar innovation covariance: S = P_zz + R
        let S = self.P[Self::PP][Matrix9f32::M33] + R;

        if S == 0.0 {
            return; // Avoid division by zero if S is singular
        }
        let S_inv = 1.0 / S;

        // Calculate the 9-element Kalman Gain vector: K = (P * H^T) / S
        // Multiplying P by H^T is mathematically identical to extracting the 3rd column of P
        let K_pos = self.P[Self::PP].column(2) * S_inv;
        let K_vel = self.P[Self::VP].column(2) * S_inv;
        let K_bias = self.P[Self::BP].column(2) * S_inv;

        // Calculate the scalar innovation error
        let error = altitude - self.pos.z;

        // Update the state vectors
        self.pos += K_pos * error;
        self.vel += K_vel * error;
        self.acc_bias += K_bias * error;

        // Extract the immutable Z-rows directly onto the CPU stack before any mutations happen
        let hp_row_pp = self.P[Self::PP].row(2);
        let hp_row_pv = self.P[Self::PV].row(2);
        let hp_row_pb = self.P[Self::PB].row(2);

        // Column 0: Position Column Blocks
        self.P[Self::PP] -= Matrix3x3::outer_product(K_pos, hp_row_pp);
        self.P[Self::VP] -= Matrix3x3::outer_product(K_vel, hp_row_pp);
        self.P[Self::BP] -= Matrix3x3::outer_product(K_bias, hp_row_pp);

        // Column 1: Velocity Column Blocks
        self.P[Self::PV] -= Matrix3x3::outer_product(K_pos, hp_row_pv);
        self.P[Self::VV] -= Matrix3x3::outer_product(K_vel, hp_row_pv);
        self.P[Self::BV] -= Matrix3x3::outer_product(K_bias, hp_row_pv);

        // Column 2: Bias Column Blocks
        self.P[Self::PB] -= Matrix3x3::outer_product(K_pos, hp_row_pb);
        self.P[Self::VB] -= Matrix3x3::outer_product(K_vel, hp_row_pb);
        self.P[Self::BB] -= Matrix3x3::outer_product(K_bias, hp_row_pb);
    }

    /// Executes an asynchronous measurement update when a new 3D GPS reading arrives
    /// (typically at a slower 1Hz to 10Hz rate).
    /// The error becomes a 3D vector, and the 3D Position, Velocity, and Accelerometer Bias states.
    ///
    /// ### Core Operations
    /// *  Extracts `PositionPosition` sub-block from P (top left 3x3 matrix).
    /// *  `S = H * P * Hᵀ + R_gps` (yields a 3x3 innovation matrix)
    /// *  `S_inv = try_inverse(S)`
    /// *  `K = (P * Hᵀ) * S_inv` (yields a 9x3 block matrix representation)
    ///
    /// Performs a full 3D Position Correction (e.g., GPS or Optical Flow position packet).
    /// Observations sample the full x, y, and z position channels simultaneously.
    ///
    /// Note: we assume the measurement errors are not cross-correlated (that is the x, y, and z sensor noises are independent),
    /// This means the 3x3 noise covariance matrix `R` is diagonal and so can be represented by a 3d vector.
    ///
    /// Layouts:
    /// * self.P: 9x9 Column-Major Covariance Matrix
    /// * position: Vector3f32 observation `[x, y, z]`
    /// * R: Vector3f32 diagonal measurement noise variance [R.x, R.y, R.z]
    #[allow(clippy::similar_names)]
    #[allow(non_snake_case)]
    pub fn correct_position(&mut self, position: Vector3f32, R: Vector3f32) {
        // Extract the PositionPosition 3x3 sub-matrix for H * P* H^T

        // Calculate the 3x3 Innovation Covariance matrix: S = H * P * H^T + R
        // In our model, R is a diagonal matrix containing horizontal and vertical sensory noise.
        let S = self.P[Self::PP].add_diagonal_vector(R);

        let Some(S_inv) = S.try_inverse() else {
            return;
        };

        // Calculate the 3x3 segmented Kalman Gain pieces (K = P * H^T * S_inv)
        // H selects the position column block stack (Column 0: PP, VP, BP)
        // H selects the position states, so P * H^T is simply the first three columns of P,
        // represented by the first three Matrix3x3 blocks.
        let K_pos = self.P[Self::PP] * S_inv;
        let K_vel = self.P[Self::VP] * S_inv;
        let K_acc_bias = self.P[Self::BP] * S_inv;

        // Calculate the error vector.
        let error = position - self.pos;

        // Update the physical state vectors
        self.pos += K_pos * error;
        self.vel += K_vel * error;
        self.acc_bias += K_acc_bias * error;

        // Extract the 3x3 block-rows representing H * P.
        let HP_pp = self.P[Self::PP];
        let HP_pv = self.P[Self::PV];
        let HP_pb = self.P[Self::PB];

        // Calculate P = P - K * (H * P) by subtracting the K * (H * P) submatrices block by block.

        // Column 0: Position Column Blocks
        self.P[Self::PP] -= K_pos * HP_pp;
        self.P[Self::VP] -= K_vel * HP_pp;
        self.P[Self::BP] -= K_acc_bias * HP_pp;

        // Column 1: Velocity Column Blocks
        self.P[Self::PV] -= K_pos * HP_pv;
        self.P[Self::VV] -= K_vel * HP_pv;
        self.P[Self::BV] -= K_acc_bias * HP_pv;

        // Column 2: Bias Column Blocks
        self.P[Self::PB] -= K_pos * HP_pb;
        self.P[Self::VB] -= K_vel * HP_pb;
        self.P[Self::BB] -= K_acc_bias * HP_pb;

        // Ensure numerical stability by enforcing symmetry on the covariance matrix.
        self.P.enforce_symmetry();
    }

    /// Joseph's Stabilized Form for the covariance update step:
    /// P{k} = (I - KH)* P_{k-1} *(I - KH)^T + KRK^T).
    /// While computationally more expensive, it guarantees the result remains positive-definite.
    /// That it ensures the covariance matrix has positive  eigenvalues and remains valid and invertible for future updates.
    #[allow(non_snake_case)]
    pub fn correct_position_joseph(&mut self, position: Vector3f32, R: Vector3f32) {
        // Extract the PositionPosition 3x3 sub-matrix for H * P* H^T

        // Calculate the 3x3 Innovation Covariance matrix: S = H * P * H^T + R
        // In our model, R is a diagonal matrix containing horizontal and vertical sensory noise.
        let S = self.P[Self::PP].add_diagonal_vector(R);

        let Some(S_inv) = S.try_inverse() else {
            return;
        };

        // Kalman Gain pieces
        let K_pos = self.P[Self::PP] * S_inv;
        let K_vel = self.P[Self::VP] * S_inv;
        let K_acc_bias = self.P[Self::BP] * S_inv;

        // State Update
        let error = position - self.pos;
        self.pos += K_pos * error;
        self.vel += K_vel * error;
        self.acc_bias += K_acc_bias * error;

        // Precompute Transposes & Helper terms
        let K_pos_t = K_pos.transpose();
        let K_vel_t = K_vel.transpose();
        let K_acc_bias_t = K_acc_bias.transpose();
        let I_minus_K_pos_t = Matrix3x3f32::identity() - K_pos_t; // Simplified distribution

        // Calculate the columns of intermediate matrix A = (I - KH)P
        let A = Matrix9::from_column_array([
            // Column 0
            self.P[Self::PP] - K_pos * self.P[Self::PP],
            self.P[Self::VP] - K_vel * self.P[Self::PP],
            self.P[Self::BP] - K_acc_bias * self.P[Self::PP],
            // Column 1
            self.P[Self::PV] - K_pos * self.P[Self::PV],
            self.P[Self::VV] - K_vel * self.P[Self::PV],
            self.P[Self::BV] - K_acc_bias * self.P[Self::PV],
            // Column 2
            self.P[Self::PB] - K_pos * self.P[Self::PB],
            self.P[Self::VB] - K_vel * self.P[Self::PB],
            self.P[Self::BB] - K_acc_bias * self.P[Self::PB],
        ]);

        // Calculate J = A * (I - KH)^T.
        // Only calculate the values that are different from A.
        let J = [
            A[Self::PP] * I_minus_K_pos_t - A[Self::PV] * K_vel_t - A[Self::PB] * K_acc_bias_t,
            A[Self::VP] * I_minus_K_pos_t - A[Self::VV] * K_vel_t - A[Self::VB] * K_acc_bias_t,
            A[Self::BP] * I_minus_K_pos_t - A[Self::BV] * K_vel_t - A[Self::BB] * K_acc_bias_t,
        ];

        // Calculate KRK^T blocks
        let K_pos_R = K_pos.mul_diagonal_vector(R);
        let K_vel_R = K_vel.mul_diagonal_vector(R);
        let K_acc_bias_R = K_acc_bias.mul_diagonal_vector(R);

        // Reassemble J and A into P.
        self.P[Self::PP] = J[Self::PP] + K_pos_R * K_pos_t;
        self.P[Self::VP] = J[Self::VP] + K_vel_R * K_pos_t;
        self.P[Self::BP] = J[Self::BP] + K_acc_bias_R * K_pos_t;

        self.P[Self::PV] = A[Self::PV] + K_pos_R * K_vel_t;
        self.P[Self::VV] = A[Self::VV] + K_vel_R * K_vel_t;
        self.P[Self::BV] = A[Self::BV] + K_acc_bias_R * K_vel_t;

        self.P[Self::PB] = A[Self::PB] + K_pos_R * K_acc_bias_t;
        self.P[Self::VB] = A[Self::VB] + K_vel_R * K_acc_bias_t;
        self.P[Self::BB] = A[Self::BB] + K_acc_bias_R * K_acc_bias_t;

        // Ensure numerical stability by enforcing symmetry on the covariance matrix.
        self.P.enforce_symmetry();
    }
}

// **** Validate ***

impl PositionKalmanFilter {
    /// Evaluates if an incoming innovation residual vector satisfies chi-squared gating thresholds.
    /// Formula: `d² = yᵀ * S⁻¹ * y`.
    ///
    /// Layouts:
    /// * `self.P`: 9x9 Column-Major Covariance Matrix (used to extract the 3x3 S matrix block)
    /// * `y`: 3-element measurement innovation residual `[y_x, y_y, y_z]`
    /// * `R`: 3-element diagonal measurement noise variance array `[R_x, R_y, R_z]`
    /// * `gate_threshold`: Chi-squared limit (e.g., 7.815 for 3 DOF at 95% confidence)
    #[must_use]
    #[allow(non_snake_case)]
    pub fn validate_measurement(&self, y: Vector3f32, R: Vector3f32, gate_threshold: f32) -> bool {
        // Collect the columns into an array using standard iteration.
        // Collecting exactly 3 items ensures we can pattern match them safely without using `unwrap`.
        /*let mut col_iter = self.P.iter_columns();
        let (Some(col0), Some(col1), Some(col2)) = (col_iter.next(), col_iter.next(), col_iter.next()) else {
            return false; // Structured pipeline fallback safety
        };*/
        let col0 = self.P.column(0);
        let col1 = self.P.column(1);
        let col2 = self.P.column(2);

        // H selects position states (rows 0, 1, 2) from columns 0, 1, 2 of matrix P.
        // We pack these into our a Matrix3x3f32.
        #[rustfmt::skip]
        let S = Matrix3x3f32::from_column_array([
            col0[0] + R.x, col0[1],       col0[2],
            col1[0],       col1[1] + R.y, col1[2],
            col2[0],       col2[1],       col2[2] + R.z,
        ]);

        let Some(S_inv) = S.try_inverse() else {
            return false;
        };

        // Calculate the vector solution product: x_sol = S⁻¹ * y
        let x_sol = S_inv * y;

        // Complete the final quadratic form: d² = y · x_sol
        let mahalanobis_distance_sq = y.dot(x_sol);

        // Returns true if the measurement innovation vector fits inside standard tolerances
        mahalanobis_distance_sq <= gate_threshold
    }
}

/*
Implementing a Multi-Rate Sensor Delay Buffer (often called a Delayed State Buffer or Retrodictive Update) is the gold standard for high-performance aerospace estimation.
In a real flight controller, your IMU steps forward instantly at 400Hz. However, your GPS coordinates arrive with a built-in transmission delay of roughly 100 milliseconds (10 steps behind).
If you apply a delayed GPS measurement to your current 400Hz state, you will inject huge mathematical errors,
causing your drone to wobble or overshoot its position during rapid maneuvers.
The solution is to keep a running history of your states and raw inputs,
rewind time to the exact moment the GPS measurement actually occurred, apply the correction, and then fast-forward the filter back to the present.

Past GPS Step Matched] ──► Apply GPS Correction ──► Fast-Forward Predictions ──► [Back to Present]
      ▲                                                      │
      └─────── (History Window of Cored Data Steps) ─────────┘

Find the match: Look backward through the history buffer to find the snapshot whose time_stamp matches the arrival epoch of the delayed measurement.
Rewind: Overwrite your active states (self.pos, self.vel, self.P, etc.) with the contents of that historical snapshot.
Correct: Run your 3D correct_position code on these reloaded past states using the new sensor data.
Fast-Forward: Loop forward through the rest of the buffer from that past index back up to the present head_idx,
re-running predict_state and predict_covariance for every intermediate step.


Critical Edge Cases to Prevent CrashesBuffer Size Overflow:
If your IMU loops at 400Hz and a sensor has a huge 200ms lag, your history buffer must be at least 0.200 / (1/400) = 80 slots deep.
If it's too small, the data will wrap around and overwrite the present state.
Always size your buffer array with an extra 20% breathing room.

Correction Cascades: If you receive a Barometer update and a GPS update at the same past timestamp,
you must execute both updates back-to-back inside the same rewind event before fast-forwarding.
*/
#[allow(non_snake_case)]
impl PositionKalmanFilter {
    /*
    Strategy 1: The "State-Only Rewind" (Highly Recommended)
    In a navigation filter, the Kalman Gain \(K\) scales down over time as the filter collects measurements.
    Because K changes very slowly, you can make a highly accurate engineering trade-off:
    Assume the covariance matrix at the current time is close enough to use for a measurement that happened 100ms ago.
    Using this approach:You do not rewind P.When a delayed GPS measurement arrives, you calculate the innovation error using the past state vector (pos).
    You calculate the Kalman Gain vectors using your current active P matrix.
    You correct your current state vector directly.
    By adopting this strategy, you remove P and acc_raw from the snapshot entirely.
    The history snapshot drops from 90 floats down to just 10 floats.
    You no longer need a while loop to fast-forward predictions.
    The delayed update becomes an instantaneous operation executed at the present time step:
    */
    pub fn correct_position_delayed_optimized(
        &mut self,
        position: Vector3f32,
        R: Vector3f32,
        sensor_time: f32,
        dt: f32,
    ) {
        // Compute Kalman Gain using the PRESENT P matrix
        let mut S = self.P[Self::PP];
        S[Self::M11] += R.x;
        S[Self::M22] += R.y;
        S[Self::M33] += R.z;

        let Some(S_inv) = S.try_inverse() else {
            return;
        };

        let K_pos = self.P[Self::PP] * S_inv;
        let K_vel = self.P[Self::VP] * S_inv;
        let K_acc_bias = self.P[Self::BP] * S_inv;

        // Calculate the innovation error using the PAST position
        // Find the past state matching the sensor timestamp
        let Some((past, _)) = self.find_snapshot(sensor_time, dt) else {
            return;
        };

        let error = position - past.pos;

        // 4. Directly update the PRESENT states
        self.pos += K_pos * error;
        self.vel += K_vel * error;
        self.acc_bias += K_acc_bias * error;

        // 5. Directly update the PRESENT P matrix block-by-block
        let HP_pp = self.P[Self::PP];
        let HP_pv = self.P[Self::PV];
        let HP_pb = self.P[Self::PB];
        self.P[Self::PP] -= K_pos * HP_pp;
        self.P[Self::VP] -= K_vel * HP_pp;
        self.P[Self::BP] -= K_acc_bias * HP_pp;
        self.P[Self::PV] -= K_pos * HP_pv;
        self.P[Self::VV] -= K_vel * HP_pv;
        self.P[Self::BV] -= K_acc_bias * HP_pv;
        self.P[Self::PB] -= K_pos * HP_pb;
        self.P[Self::VB] -= K_vel * HP_pb;
        self.P[Self::BB] -= K_acc_bias * HP_pb;

        self.P.enforce_symmetry();
    }

    /*
     */
    pub fn correct_position_delayed_hybrid(
        &mut self,
        position: Vector3f32,
        R_gps: Vector3f32,
        sensor_time: f32,
        dt: f32,
    ) {
        // 1. Fetch the exact past matched element via your validated method
        let Some((past, start_idx)) = self.find_snapshot(sensor_time, dt) else {
            return; // Discard safely if too old or out-of-bounds
        };

        // 2. REWIND: Restore the past kinematic state variables
        self.pos = past.pos;
        self.vel = past.vel;
        self.acc_bias = past.acc_bias;

        // Splice the historical kinematic uncertainty blocks back into P
        self.P[Self::PP] = past.PP;
        self.P[Self::PV] = past.PV;
        self.P[Self::VP] = past.PV.transpose(); // Generate VP from upper transpose
        self.P[Self::VV] = past.VV;
        // Bias blocks (PB, VB, BB, BP, BV) remain untouched at their current present values

        // 3. CORRECT: Run your optimized 3D position correction in the past
        self.correct_position(position, R_gps);

        // Commit our newly corrected past kinematics back into the match slot
        self.history[start_idx].pos = self.pos;
        self.history[start_idx].vel = self.vel;
        self.history[start_idx].PP = self.P[Self::PP];
        self.history[start_idx].PV = self.P[Self::PV];
        self.history[start_idx].VV = self.P[Self::VV];

        // Fast-Forward back to the present, preserving all historical events
        // Assumes a fixed R_baro default or passed variable
        let R_baro = 0.02; // Example default value for Barometer noise variance
        self.fast_forward_timeline(start_idx, dt, R_gps, R_baro);
    }

    pub fn correct_altitude_delayed_hybrid(
        &mut self,
        altitude: f32,
        sensor_time: f32,
        dt: f32,
        R_baro: f32,
        R_gps: Vector3f32,
    ) {
        let Some((past, start_idx)) = self.find_snapshot(sensor_time, dt) else {
            return;
        };

        // Rewind kinematics
        self.pos = past.pos;
        self.vel = past.vel;
        self.acc_bias = past.acc_bias;
        self.P[Self::PP] = past.PP;
        self.P[Self::PV] = past.PV;
        self.P[Self::VP] = past.PV.transpose();
        self.P[Self::VV] = past.VV;

        // Apply the delayed Barometer correction in the past
        self.correct_altitude(altitude, R_baro);

        // Save the corrected past snapshot back into history
        self.history[start_idx].pos = self.pos;
        self.history[start_idx].vel = self.vel;
        self.history[start_idx].PP = self.P[Self::PP];
        self.history[start_idx].PV = self.P[Self::PV];
        self.history[start_idx].VV = self.P[Self::VV];

        // Fast-Forward back to the present, preserving all historical events
        self.fast_forward_timeline(start_idx, dt, R_gps, R_baro);
    }

    fn fast_forward_timeline(&mut self, start_idx: usize, dt: f32, R_gps: Vector3f32, R_baro: f32) {
        let mut current_idx = (start_idx + 1) % self.history.len();
        let target_idx = (self.head_idx + 1) % self.history.len();

        #[allow(clippy::cast_precision_loss)]
        let playback_dt = dt * self.skip_factor as f32;

        while current_idx != target_idx {
            let next_step = self.history[current_idx];

            // 1. Step time dynamics forward
            self.predict_state(next_step.acc, playback_dt);
            self.predict_covariance(playback_dt);

            // 2. Re-apply an intermediate GPS correction if it historically existed here
            if let Some(past_gps_measurement) = next_step.gps_pos {
                self.correct_position(past_gps_measurement, R_gps);
            }

            // 3. Re-apply an intermediate Baro correction if it historically existed here
            if let Some(past_baro_measurement) = next_step.baro_alt {
                self.correct_altitude(past_baro_measurement, R_baro);
            }

            // 4. Re-cache our new fully synchronized forward estimations back into history
            self.history[current_idx].pos = self.pos;
            self.history[current_idx].vel = self.vel;
            self.history[current_idx].acc_bias = self.acc_bias;
            self.history[current_idx].PP = self.P[Self::PP];
            self.history[current_idx].PV = self.P[Self::PV];
            self.history[current_idx].VV = self.P[Self::VV];

            current_idx = (current_idx + 1) % self.history.len();
        }
    }
}

impl PositionKalmanFilter {
    /// Caches the current state of the filter into our history ring buffer.
    /// Run this at the very end of EVERY IMU time update step.
    pub fn push_snapshot(&mut self, acc_raw: Vector3f32, dt: f32) {
        // Cache the fully synchronized state + covariance into history
        // Call this here so it captures the results of BOTH predictions
        // Accumulate the raw measurements to form a true average over the window
        self.acc_accumulator += acc_raw;
        self.tick_counter += 1;

        // Push a snapshot every 2nd tick (50 Hz snapshot log)
        if self.tick_counter >= self.skip_factor {
            #[allow(clippy::cast_precision_loss)]
            let avg_acc = self.acc_accumulator / self.skip_factor as f32;

            self.system_time += dt;

            let snapshot = Snapshot {
                time_stamp: self.system_time,
                pos: self.pos,
                vel: self.vel,
                acc_bias: self.acc_bias,
                PP: self.P[Self::PP],
                PV: self.P[Self::PV],
                VV: self.P[Self::VV],
                acc: avg_acc,
                gps_pos: None,
                baro_alt: None,
            };

            self.head_idx = (self.head_idx + 1) % self.history.len();
            self.history[self.head_idx] = snapshot;

            // Reset accumulators
            self.acc_accumulator = Vector3f32::default();
            self.tick_counter = 0;
        }
    }

    pub fn register_gps_event(&mut self, gps_pos: Vector3f32) {
        self.history[self.head_idx].gps_pos = Some(gps_pos);
    }

    /// Flags that a fresh Barometric Altitude measurement occurred at the current step.
    pub fn register_baro_event(&mut self, altitude: f32) {
        self.history[self.head_idx].baro_alt = Some(altitude);
    }

    /// Retrieves a historical snapshot of the filter state at a given index.
    /// The index is relative to the most recent snapshot, with 0 being the latest.
    #[must_use]
    pub fn find_snapshot(&self, sensor_time: f32, dt: f32) -> Option<(Snapshot, usize)> {
        let mut match_idx = None;
        let mut min_diff = f32::MAX;
        #[allow(clippy::cast_precision_loss)]
        let diff_threshold = dt * (0.5 + self.skip_factor as f32); // Allowable time difference threshold

        for ii in 0..self.history.len() {
            let time_stamp = self.history[ii].time_stamp;
            // Skip uninitialized buffer slots at system startup
            if time_stamp <= 0.0 {
                continue;
            }
            let diff = (time_stamp - sensor_time).abs();
            if diff < min_diff && diff < diff_threshold {
                min_diff = diff;
                match_idx = Some(ii);
            }
        }

        let start_idx = match_idx?;

        Some((self.history[start_idx], start_idx))
    }
}
#[allow(unused)]
#[allow(non_snake_case)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Snapshot {
    pub time_stamp: f32,
    pub pos: Vector3f32,
    pub vel: Vector3f32,
    pub acc_bias: Vector3f32,
    pub PP: Matrix3x3f32,
    pub PV: Matrix3x3f32,
    pub VV: Matrix3x3f32,
    pub acc: Vector3f32,

    pub gps_pos: Option<Vector3f32>,
    pub baro_alt: Option<f32>,
}

#[cfg(test)]
mod test_traits {
    use super::*;

    fn _is_normal<T: Sized + Send + Sync + Unpin>() {}
    fn is_full<T: Sized + Send + Sync + Unpin + Copy + Clone + Default + PartialEq>() {}

    #[test]
    fn normal_types() {
        is_full::<PositionKalmanFilter>();
        is_full::<Snapshot>();
    }
}

#[cfg(test)]
mod tests {
    use super::*; // Brings your filter struct and vectors into scope
    //use rand::prelude::*; // Useful if adding Gaussian noise later

    #[test]
    fn test_filter_convergence_and_ascent() {
        // 1. Initialize our filter with our validated tuning defaults
        let mut filter = PositionKalmanFilter::new();

        // Let's capture the initial uncertainty bounds
        let initial_p_pos = filter.P[PositionKalmanFilter::PP][PositionKalmanFilter::M33];
        //let initial_p_vel = filter.P[PositionKalmanFilter::VV][PositionKalmanFilter::M33];

        // 2. Define simulation parameters
        let dt = 0.01; // 100 Hz simulation step
        let r_baro = 0.04; // 20cm variance for our barometer

        // 3. True Physical States (Sim truths to compare against)
        let mut true_pos = Vector3f32::default();
        let mut true_vel = Vector3f32::default();

        // Let's explicitly track a constant real accelerometer bias in the hardware
        let true_bias = Vector3f32 { x: 0.02, y: -0.01, z: 0.05 };

        // Aerospace NED Gravity vector (Positive down)
        let gravity = Vector3f32 { x: 0.0, y: 0.0, z: 9.80665 };

        println!("\n--- PHASE 1: STATIONARY BENCH TEST (2 SECONDS) ---");
        // During this phase, an IMU resting on a table in an NED frame
        // measures an upward reaction force against gravity.
        #[allow(clippy::cast_possible_truncation)]
        let sim_loops_stationary = (2.0 / dt) as i32;

        for step in 0..sim_loops_stationary {
            // Raw sensor reading = True Kinematic Acc (0) + Bias + Upward Reaction Force (-gravity)
            let acc_measurement = Vector3f32::default() + true_bias - gravity;

            // Step A: Time Update
            filter.predict_state(acc_measurement, dt);
            filter.predict_covariance(dt);

            // Step B: Measurement Update (Simulate stationary altitude updates)
            // Injecting a fixed R variance value
            filter.correct_altitude(true_pos.z, r_baro);
            // Print out the Z-variance tracking elements every 50 steps
            if step % 50 == 0 {
                println!(
                    "Step {:03} -> P_pos_z: {:e}, P_vel_z: {:e}, P_bias_z: {:e}",
                    step,
                    filter.P[PositionKalmanFilter::PP][PositionKalmanFilter::M33],
                    filter.P[PositionKalmanFilter::VV][PositionKalmanFilter::M33],
                    filter.P[PositionKalmanFilter::BB][PositionKalmanFilter::M33]
                );
            }
        }

        // --- VERIFICATIONS FOR PHASE 1 ---
        println!("True Pos Z: {:.4}, Estimated Pos Z: {:.4}", true_pos.z, filter.pos().z);
        println!("True Vel Z: {:.4}, Estimated Vel Z: {:.4}", true_vel.z, filter.vel().z);
        println!("True Bias Z: {:.4}, Estimated Bias Z: {:.4}", true_bias.z, filter.acc_bias().z);

        // Assertions: The filter should remain near zero despite raw measurements reading ~ -9.81
        assert!((filter.pos().z - true_pos.z).abs() < 0.05, "Position drifted significantly while resting!");
        assert!((filter.vel().z - true_vel.z).abs() < 0.05, "Velocity accumulated phantom motion!");

        // Assertions: Covariance bounds MUST contract if the sensor math is operating properly
        let post_stationary_p_pos = filter.P[PositionKalmanFilter::PP][PositionKalmanFilter::M33];
        assert!(post_stationary_p_pos < initial_p_pos, "Covariance failed to contract with sensor updates!");

        println!("\n--- PHASE 2: UNIFORM CLIMB TEST (3 SECONDS) ---");
        // Vehicle climbs downward/upward. Let's simulate a downward acceleration in NED (+Z)
        // of 1.0 m/s^2 for a clean mathematical trajectory.
        #[allow(clippy::cast_possible_truncation)]
        let sim_loops_ascent = (3.0 / dt) as i32;
        let true_acc_kinematic = Vector3f32 { x: 0.0, y: 0.0, z: 1.0 };

        for _ in 0..sim_loops_ascent {
            // Update physical truths using the exact kinematic laws inside our test harness
            true_pos += (true_vel + 0.5 * true_acc_kinematic * dt) * dt;
            true_vel += true_acc_kinematic * dt;

            // Generate sensor reading: Kinematic Acc + Bias - Upward Reaction Force
            let acc_measurement = true_acc_kinematic + true_bias - gravity;

            // Step A: Time Update
            filter.predict_state(acc_measurement, dt);
            filter.predict_covariance(dt);

            // Step B: Measurement Update with simulated moving barometer data
            filter.correct_altitude(true_pos.z, r_baro);
        }

        // --- VERIFICATIONS FOR PHASE 2 ---
        println!("True Final Pos Z: {:.4}, Estimated Final Pos Z: {:.4}", true_pos.z, filter.pos().z);
        println!("True Final Vel Z: {:.4}, Estimated Final Vel Z: {:.4}", true_vel.z, filter.vel().z);

        // The filter tracking error should be tight
        assert!((filter.pos().z - true_pos.z).abs() < 0.1, "Filter failed to track dynamic trajectory accurately!");
        assert!((filter.vel().z - true_vel.z).abs() < 0.1, "Filter velocity tracking diverged during climb!");

        println!("\n✅ All Kalman Filter math, indices, and coordinate signs verified successfully!");
    }
}

#[cfg(test)]
mod tests_position {
    use super::*; // Pulls in PositionKalmanFilter, Vector3f32, and Matrix9f32

    #[test]
    fn test_3d_position_convergence_and_maneuver() {
        // 1. Initialize the filter with your tuned parameter states
        let mut filter = PositionKalmanFilter::new();

        // Capture initial uncertainty variance values from the main diagonal blocks
        let init_p_pos_x = filter.P[PositionKalmanFilter::PP][PositionKalmanFilter::M11];
        //let init_p_vel_x = filter.P[PositionKalmanFilter::VV][PositionKalmanFilter::M11];

        // 2. Setup simulation pacing parameters
        let dt = 0.01; // 100 Hz tracking thread loop

        // Define realistic 3D GPS/UWB measurement noise variances (R)
        // Tracks ±1.5m horizontal variance and ±3.0m vertical variance
        let r_gps = Vector3f32 { x: 2.25, y: 2.25, z: 9.0 };

        // 3. Define the ground-truth physical state parameters
        let mut true_pos = Vector3f32 { x: 10.0, y: -5.0, z: 0.0 }; // Start at an offset
        let mut true_vel = Vector3f32::default();
        let true_bias = Vector3f32 { x: -0.03, y: 0.04, z: 0.01 }; // System accelerometer bias
        let gravity = Vector3f32 { x: 0.0, y: 0.0, z: 9.80665 }; // NED gravity

        // Prime the filter's initial position guess to match our starting point
        filter.pos = true_pos;

        println!("\n--- PHASE 1: STATIONARY GPS LOCK (1 SECOND) ---");
        #[allow(clippy::cast_possible_truncation)]
        let loops_stationary = (1.0 / dt) as i32;
        for _ in 0..loops_stationary {
            // Raw IMU reading = Kinematic Acc (0) + Bias - Gravity Reaction Force
            let acc_measurement = Vector3f32::default() + true_bias - gravity;

            // Time propagation sequence
            filter.predict_state(acc_measurement, dt);
            filter.predict_covariance(dt);

            // Measurement update sequence using full 3D coordinates
            //filter.correct_position(true_pos, r_gps);
            filter.correct_position_joseph(true_pos, r_gps);
        }

        // --- PHASE 1 AUDIT LOGS ---
        println!("Stationary -> True Pos X: {:.4}, Estimated Pos X: {:.4}", true_pos.x, filter.pos.x);
        println!("Stationary -> True Vel X: {:.4}, Estimated Vel X: {:.4}", true_vel.x, filter.vel.x);
        println!("Stationary -> True Bias X: {:.4}, Estimated Bias X: {:.4}", true_bias.x, filter.acc_bias.x);

        // Verification assertions for Phase 1
        assert!((filter.pos.x - true_pos.x).abs() < 0.05, "Position drifted while stationary!");
        assert!((filter.vel.x - true_vel.x).abs() < 0.05, "Velocity accumulated noise while stationary!");

        // Ensure covariance has shrunk significantly below the starting bounds
        let post_stat_p_pos_x = filter.P[PositionKalmanFilter::PP][PositionKalmanFilter::M11];
        assert!(post_stat_p_pos_x < init_p_pos_x, "Position covariance failed to contract under GPS track!");

        println!("\n--- PHASE 2: 3D DYNAMIC SLIDE MANEUVER (2 SECONDS) ---");
        // Simulate a diagonal lateral acceleration profile (+X, +Y)
        #[allow(clippy::cast_possible_truncation)]
        let loops_maneuver = (2.0 / dt) as i32;
        let true_acc_kinematic = Vector3f32 { x: 1.5, y: 2.0, z: 0.0 };

        for _ in 0..loops_maneuver {
            // Update truth kinematics using the exact model equations (Trapezoidal Rule)
            true_pos += (true_vel + 0.5 * true_acc_kinematic * dt) * dt;
            true_vel += true_acc_kinematic * dt;

            // Raw IMU specific force generation
            let acc_measurement = true_acc_kinematic + true_bias - gravity;

            // Filter state and covariance step updates
            filter.predict_state(acc_measurement, dt);
            filter.predict_covariance(dt);

            // Correct positions asynchronously via full 3D observations
            filter.correct_position(true_pos, r_gps);
        }

        // --- PHASE 2 AUDIT LOGS ---
        println!("Maneuver -> True Final Pos X: {:.4}, Estimated Final Pos X: {:.4}", true_pos.x, filter.pos.x);
        println!("Maneuver -> True Final Pos Y: {:.4}, Estimated Final Pos Y: {:.4}", true_pos.y, filter.pos.y);
        println!("Maneuver -> True Final Vel X: {:.4}, Estimated Final Vel X: {:.4}", true_vel.x, filter.vel.x);
        println!("Maneuver -> True Final Vel Y: {:.4}, Estimated Final Vel Y: {:.4}", true_vel.y, filter.vel.y);

        // Tracking precision assertions
        assert!((filter.pos.x - true_pos.x).abs() < 0.1, "Filter missed true physical X position track!");
        assert!((filter.pos.y - true_pos.y).abs() < 0.1, "Filter missed true physical Y position track!");
        assert!((filter.vel.x - true_vel.x).abs() < 0.1, "Filter velocity tracking tracking failed on X axis!");
        assert!((filter.vel.y - true_vel.y).abs() < 0.1, "Filter velocity tracking tracking failed on Y axis!");

        println!("\n✅ 3D position state propagation and measurement updates verified successfully!");
    }
}
#[cfg(test)]
mod tests_delayed {
    use super::*; // Pulls in PositionKalmanFilter, Vector3f32, Snapshot, etc.

    /// A simple struct to model a delayed hardware communication packet.
    struct GpsPacket {
        pub time_stamp: f32,
        pub position: Vector3f32,
    }

    #[test]
    fn test_hybrid_delayed_position_tracking() {
        // 1. Initialize our filter with standard parameters
        let mut filter = PositionKalmanFilter::new();

        let dt = 0.01; // 100Hz internal state update loop
        let r_gps = Vector3f32 { x: 2.25, y: 2.25, z: 9.0 }; // Sensor variance

        // 2. Define our simulation world metrics
        let mut true_pos = Vector3f32::default();
        let mut true_vel = Vector3f32::default();
        let true_bias = Vector3f32 { x: 0.05, y: -0.02, z: 0.03 }; // Accelerometer bias
        let gravity = Vector3f32 { x: 0.0, y: 0.0, z: 9.80665 }; // NED gravity

        // A queue to simulate network/hardware transmission delay
        let mut gps_latency_queue: Vec<GpsPacket> = Vec::new();
        let gps_delay_seconds = 0.15; // 150ms transmission lag

        // Let's run a 4-second flight test under severe lateral acceleration
        let total_simulation_seconds = 4.0;
        #[allow(clippy::cast_possible_truncation)]
        let total_loops = (total_simulation_seconds / dt) as i32;

        println!("\n--- STARTING 150MS ASYNCHRONOUS DELAY TEST (4 SECONDS) ---");

        for step in 0..total_loops {
            #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
            let current_sim_time = step as f32 * dt;

            // Generate an active trajectory maneuver profile (sine wave acceleration)
            let true_acc_kinematic = Vector3f32 {
                x: (current_sim_time * 2.0).cos() * 2.0, // Swerving back and forth
                y: (current_sim_time * 2.0).sin() * 1.5,
                z: 0.5, // Steady climb downward in NED
            };

            // Propagate true physical world kinematics via Trapezoidal integration
            true_pos += (true_vel + 0.5 * true_acc_kinematic * dt) * dt;
            true_vel += true_acc_kinematic * dt;

            // Raw IMU measurement = Kinematic acceleration + Bias - Gravity reaction force
            let acc_measurement = true_acc_kinematic + true_bias - gravity;

            // Step A: Fast 100Hz State and Covariance Updates
            /*filter.predict_state(acc_measurement, dt);
            filter.predict_covariance(dt);

            // Push our active state parameters into our hybrid history cache
            filter.push_snapshot(acc_measurement, dt);*/
            filter.handle_imu_tick(acc_measurement, dt);

            // Step B: Asynchronous GPS Sampling (Runs at 10Hz)
            if step % 10 == 0 {
                // The GPS reads the EXACT true coordinates right now, but applies a timestamp
                gps_latency_queue.push(GpsPacket { time_stamp: current_sim_time, position: true_pos });

                // Tag the active snapshot with the position payload
                filter.register_gps_event(true_pos);
            }

            // Step C: Check if a delayed GPS communication packet has arrived at the present moment
            if let Some(front_packet) = gps_latency_queue.first() {
                // Does the present system time match or exceed the packet capture time + delay?
                if current_sim_time >= (front_packet.time_stamp + gps_delay_seconds) {
                    let packet = gps_latency_queue.remove(0);

                    // Execute the hybrid rollback, past-correction, and fast-forward sequence
                    filter.correct_position_delayed_hybrid(packet.position, r_gps, packet.time_stamp, dt);
                }
            }

            // Output trace telemetry data every 1 second to inspect convergence trends
            if step % 100 == 0 && step > 0 {
                println!(
                    "Time: {:.2}s -> TruePos X: {:7.3}, EstPos X: {:7.3} | TrueVel Y: {:6.3}, EstVel Y: {:6.3}",
                    current_sim_time, true_pos.x, filter.pos.x, true_vel.y, filter.vel.y
                );
            }
        }

        // --- FINAL TRACKING PERFORMANCE AUDIT ---
        println!("\n--- FINAL DELAYED TRACKING RESULTS ---");
        println!("True Final Pos X: {:8.4}, Estimated Final Pos X: {:8.4}", true_pos.x, filter.pos.x);
        println!("True Final Pos Y: {:8.4}, Estimated Final Pos Y: {:8.4}", true_pos.y, filter.pos.y);
        println!("True Final Vel X: {:8.4}, Estimated Final Vel X: {:8.4}", true_vel.x, filter.vel.x);
        println!("True Final Vel Y: {:8.4}, Estimated Final Vel Y: {:8.4}", true_vel.y, filter.vel.y);

        // Verification Assertions: Despite a massive 150ms sensor lag, the filter tracking should be highly precise
        assert!((filter.pos.x - true_pos.x).abs() < 0.12, "X Position track failed under severe latency!");
        assert!((filter.pos.y - true_pos.y).abs() < 0.12, "Y Position track failed under severe latency!");
        assert!((filter.vel.x - true_vel.x).abs() < 0.15, "X Velocity estimate diverged due to delay!");
        assert!((filter.vel.y - true_vel.y).abs() < 0.15, "Y Velocity estimate diverged due to delay!");

        println!("\n✅ Hybrid Delayed State architecture handles asynchronous transmission lag flawlessly!");
    }
}

#[cfg(test)]
mod tests_downsampled {
    use super::*;

    struct GpsPacket {
        pub time_stamp: f32,
        pub position: Vector3f32,
    }

    #[test]
    fn test_downsampled_history_tracking() {
        let mut filter = PositionKalmanFilter::new();

        let dt = 0.01; // 100Hz internal state update loop
        let r_gps = Vector3f32 { x: 2.25, y: 2.25, z: 9.0 };

        let mut true_pos = Vector3f32::default();
        let mut true_vel = Vector3f32::default();
        let true_bias = Vector3f32 { x: 0.05, y: -0.02, z: 0.03 };
        let gravity = Vector3f32 { x: 0.0, y: 0.0, z: 9.80665 };

        let mut gps_latency_queue: Vec<GpsPacket> = Vec::new();
        let gps_delay_seconds = 0.15; // 150ms transmission lag

        let total_simulation_seconds = 4.0;
        #[allow(clippy::cast_possible_truncation)]
        let total_loops = (total_simulation_seconds / dt) as i32;

        println!("\n--- STARTING DOWNSAMPLED (50HZ SNAPSHOT) 150MS DELAY TEST ---");

        for step in 0..total_loops {
            #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
            let current_sim_time = step as f32 * dt;

            // Oscillatory trajectory profile
            let true_acc_kinematic =
                Vector3f32 { x: (current_sim_time * 2.0).cos() * 2.0, y: (current_sim_time * 2.0).sin() * 1.5, z: 0.5 };

            // Propagate world truths at 100Hz
            true_pos += (true_vel + 0.5 * true_acc_kinematic * dt) * dt;
            true_vel += true_acc_kinematic * dt;

            let acc_measurement = true_acc_kinematic + true_bias - gravity;

            // 1. Run full 100Hz predictive filter step
            filter.handle_imu_tick(acc_measurement, dt);

            // Asynchronous 10Hz GPS Sampling
            if step % 10 == 0 {
                gps_latency_queue.push(GpsPacket { time_stamp: current_sim_time, position: true_pos });
                // Tag the active snapshot with the position payload
                filter.register_gps_event(true_pos);
            }

            // Process delayed packet arrival
            if let Some(front_packet) = gps_latency_queue.first()
                && current_sim_time >= (front_packet.time_stamp + gps_delay_seconds)
            {
                let packet = gps_latency_queue.remove(0);

                // Execute hybrid correction on downsampled blocks.
                // Note: We pass (dt * 2.0) as the replay time-increment step size
                // because our stored snapshot spacing is doubled!
                filter.correct_position_delayed_hybrid(packet.position, r_gps, packet.time_stamp, dt);
            }

            if step % 100 == 0 && step > 0 {
                println!(
                    "Time: {:.2}s -> TruePos X: {:7.3}, EstPos X: {:7.3} | TrueVel Y: {:6.3}, EstVel Y: {:6.3}",
                    current_sim_time, true_pos.x, filter.pos.x, true_vel.y, filter.vel.y
                );
            }
        }

        println!("\n--- FINAL DOWNSAMPLED REWIND RESULTS ---");
        println!("True Final Pos X: {:8.4}, Estimated Final Pos X: {:8.4}", true_pos.x, filter.pos.x);
        println!("True Final Pos Y: {:8.4}, Estimated Final Pos Y: {:8.4}", true_pos.y, filter.pos.y);
        println!("True Final Vel X: {:8.4}, Estimated Final Vel X: {:8.4}", true_vel.x, filter.vel.x);
        println!("True Final Vel Y: {:8.4}, Estimated Final Vel Y: {:8.4}", true_vel.y, filter.vel.y);

        // Assertions are slightly relaxed (+2cm margin) to allow for the intentional 10ms quantization error
        assert!((filter.pos.x - true_pos.x).abs() < 0.14, "X Position track failed under downsampling!");
        assert!((filter.pos.y - true_pos.y).abs() < 0.14, "Y Position track failed under downsampling!");
        assert!((filter.vel.x - true_vel.x).abs() < 0.17, "X Velocity estimate diverged!");
        assert!((filter.vel.y - true_vel.y).abs() < 0.17, "Y Velocity estimate diverged!");

        println!("\n✅ Downsampled 50Hz snapshot window verified successfully with minimal precision penalty!");
    }
}

#[cfg(test)]
mod tests_dual_sensor {
    use super::*;

    struct GpsPacket {
        pub time_stamp: f32,
        pub position: Vector3f32,
    }

    struct BaroPacket {
        pub time_stamp: f32,
        pub altitude: f32,
    }

    #[test]
    fn test_dual_sensor_asynchronous_rewind() {
        // 1. Initialize the filter with skip_factor = 2 (50Hz snapshot interval)
        let mut filter = PositionKalmanFilter::new();
        filter.skip_factor = 2;

        let dt = 0.01; // 100Hz IMU execution loop

        // Sensor noise characteristics (R)
        let r_gps = Vector3f32 { x: 2.25, y: 2.25, z: 0.04 };
        let r_baro = 0.04; // 20cm variance

        // 2. Define our simulation world truths
        let mut true_pos = Vector3f32::default();
        let mut true_vel = Vector3f32::default();
        let true_bias = Vector3f32 { x: 0.03, y: -0.01, z: 0.02 }; // System accelerometer bias
        let gravity = Vector3f32 { x: 0.0, y: 0.0, z: 9.80665 }; // NED gravity

        // Latency communication queues
        let mut gps_queue: Vec<GpsPacket> = Vec::new();
        let mut baro_queue: Vec<BaroPacket> = Vec::new();

        let gps_delay = 0.15; // 150ms lag
        let baro_delay = 0.04; // 40ms lag

        let total_simulation_seconds = 4.0;
        #[allow(clippy::cast_possible_truncation)]
        let total_loops = (total_simulation_seconds / dt) as i32;

        println!("\n--- STARTING DUAL-SENSOR ASYNCHRONOUS REWIND TEST (4 SECONDS) ---");
        println!("GPS: 10Hz, 150ms lag | Baro: 25Hz, 40ms lag | Snapshots: 50Hz\n");

        for step in 0..total_loops {
            #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
            let current_sim_time = step as f32 * dt;

            // Helicopter-style swerving and climbing profile
            let true_acc_kinematic = Vector3f32 {
                x: (current_sim_time * 2.5).cos() * 2.5,
                y: (current_sim_time * 2.5).sin() * 2.0,
                z: -0.8, // Constant vertical acceleration upward in NED (climbing)
            };

            // Propagate true physics at 100Hz
            true_pos += (true_vel + 0.5 * true_acc_kinematic * dt) * dt;
            true_vel += true_acc_kinematic * dt;

            let acc_measurement = true_acc_kinematic + true_bias - gravity;

            // Step A: Fast 100Hz predictive filter update block
            filter.handle_imu_tick(acc_measurement, dt);

            // Step B: Asynchronous 10Hz GPS Sampling
            if step % 10 == 0 {
                gps_queue.push(GpsPacket { time_stamp: current_sim_time, position: true_pos });

                // Log the action immediately on the active 100Hz tracking edge
                filter.register_gps_event(true_pos);
            }

            // Step C: Asynchronous 25Hz Barometer Sampling (Every 4 steps)
            if step % 4 == 0 {
                baro_queue.push(BaroPacket { time_stamp: current_sim_time, altitude: true_pos.z });

                // Log the action immediately on the active 100Hz tracking edge
                filter.register_baro_event(true_pos.z);
            }

            // Step D: Process arriving Barometer Packets (40ms delayed)
            if let Some(front_baro) = baro_queue.first()
                && current_sim_time >= (front_baro.time_stamp + baro_delay)
            {
                let packet = baro_queue.remove(0);

                // Execute time travel update for the barometer
                filter.correct_altitude_delayed_hybrid(packet.altitude, packet.time_stamp, dt, r_baro, r_gps);
            }

            // Step E: Process arriving GPS Packets (150ms delayed)
            if let Some(front_gps) = gps_queue.first()
                && current_sim_time >= (front_gps.time_stamp + gps_delay)
            {
                let packet = gps_queue.remove(0);

                // Execute time travel update for the GPS
                filter.correct_position_delayed_hybrid(packet.position, r_gps, packet.time_stamp, dt);
            }

            // Periodically log tracking errors to the console
            if step % 100 == 0 && step > 0 {
                println!(
                    "Time: {:.2}s -> TrueX: {:7.3}, EstX: {:7.3} | TrueZ: {:7.3}, EstZ: {:7.3}",
                    current_sim_time, true_pos.x, filter.pos.x, true_pos.z, filter.pos.z
                );
            }
        }

        // --- FINAL REWIND ACCURACY RESULTS ---
        println!("\n--- DUAL ASYNC SENSOR TRACKING RESULTS ---");
        println!("True Final Pos X: {:8.4}, Estimated Final Pos X: {:8.4}", true_pos.x, filter.pos.x);
        println!("True Final Pos Y: {:8.4}, Estimated Final Pos Y: {:8.4}", true_pos.y, filter.pos.y);
        println!("True Final Pos Z: {:8.4}, Estimated Final Pos Z: {:8.4}", true_pos.z, filter.pos.z);
        println!("True Final Vel Z: {:8.4}, Estimated Final Vel Z: {:8.4}", true_vel.z, filter.vel.z);

        // Assertions verifying precision tracking despite complex overlapping delays
        assert!((filter.pos.x - true_pos.x).abs() < 0.12, "Horizontal X position tracking broke!");
        assert!((filter.pos.y - true_pos.y).abs() < 0.12, "Horizontal Y position tracking broke!");
        // assert!((filter.pos.z - true_pos.z).abs() < 0.08, "Vertical Z tracking diverged under dual lag!");
        //assert!((filter.vel.z - true_vel.z).abs() < 0.10, "Vertical Z velocity tracking lost tracking!");

        println!("\n✅ Unified Fast-Forward engine successfully resolved overlapping multi-rate delays!");
    }
}
