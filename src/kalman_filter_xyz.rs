use vqm::{Matrix3x3, Matrix3x3f32, Matrix3x3xM3x3, Matrix3x3xM3x3f32, Vector2f32, Vector3f32};

/// `f32` variant of `PositionKalmanFilter0`.
pub type KalmanFilterXYZf32 = KalmanFilterXYZ;

/*
Rule of Thumb for Tuning

If the filter output looks laggy or sluggish (clinging too closely to the old path and trailing behind rapid movements),
you need to trust the physics model less and the sensors more:
Action: Increase Q or Decrease R.

If the filter output looks jittery or nervous (shaking whenever the sensor outputs minor noise spikes),
you are trusting the noisy sensors too heavily over the smooth physics predictions:
Action: Decrease Q or Increase R.
*/
/// Linear Kalman Filter in 3 dimensions.
///
/// The system is split into two cleanly decoupled steps. This:
/// 1. reduces the covariance matrix from 15x15 to 9x9.
/// 2. allows a Linear Kalman Filter (rather than an Extended Kalman Filter) to be used.
///
/// ```text
///   ┌─────┐  Acc/Gyro  ┌─────────────────┐
///   │ IMU ├──────────► │ MADGWICK FILTER ├──────► Orientation Quaternion
///   └─────┘            └─────────────────┘
///  Acc │────────────────────────────────────────► Acc
///      │            ┌────────────────────────┐
///      └──────────► │ POSITION KALMAN FILTER ├──► Position Vector
///                   └────────────────────────┘    Velocity Vector
///                                │
///   GPS, Barometer etc ─────►────┘
/// ```
#[allow(non_snake_case)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KalmanFilterXYZ {
    /// 3D Kinematic State Vectors.
    pub state: KalmanStateXYZ,

    // Predicted System Uncertainty Covariance Matrix (P).
    /// `P`: Prediction error covariance (aka state covariance matrix, the system's internal uncertainty).
    pub P: Matrix3x3xM3x3f32,

    // state transition noise covariance Matrix `Q`
    /// Process Noise spectral density mapping to Velocity variance.
    pub Q_velocity: f32,
    /// Process Noise spectral density mapping to Sensor Drift variance.
    pub Q_bias: f32,
}

impl Default for KalmanFilterXYZ {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(missing_docs)]
impl KalmanFilterXYZ {
    const M11: usize = Matrix3x3xM3x3f32::M11;
    const M22: usize = Matrix3x3xM3x3f32::M22;
    const M33: usize = Matrix3x3xM3x3f32::M33;

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

impl KalmanFilterXYZ {
    /// Constructor.
    #[allow(non_snake_case)]
    #[must_use]
    pub fn new() -> Self {
        let mut P = Matrix3x3xM3x3f32::default();

        // Seed initial Position uncertainty (ie, confident within 1 meter)
        P[Self::PP][Self::M11] = 1.0;
        P[Self::PP][Self::M22] = 1.0;
        P[Self::PP][Self::M33] = 1.0;

        // Seed initial Velocity uncertainty (ie, confident within 0.5 m/s)
        P[Self::VV][Self::M11] = 0.25;
        P[Self::VV][Self::M22] = 0.25;
        P[Self::VV][Self::M33] = 0.25;

        // Seed initial Bias uncertainty (ie, accelerometer bias)
        P[Self::BB][Self::M11] = 0.01;
        P[Self::BB][Self::M22] = 0.01;
        P[Self::BB][Self::M33] = 0.01;

        Self {
            state: KalmanStateXYZ::new(),
            // A value of 0.05 implies that every second, we expect aerodynamic buffeting, vibration, or wind to naturally perturb the velocity
            // by roughly 0.22 m/s ie sqrt(0.05).
            Q_velocity: 0.05,
            // Sensor bias shifts very slowly due to thermal changes as the silicon heats up.
            // So this value should be tiny so the filter treats bias as a near-constant,
            // shifting it incrementally over minutes rather than fluctuating on every single vibration loop.
            Q_bias: 1e-4,
            P,
        }
    }
}

impl KalmanFilterXYZ {
    #[inline]
    #[must_use]
    pub fn pos(&self) -> Vector3f32 {
        self.state.pos
    }

    #[inline]
    #[must_use]
    pub fn vel(&self) -> Vector3f32 {
        self.state.vel
    }

    #[inline]
    #[must_use]
    pub fn acc_bias(&self) -> Vector3f32 {
        self.state.acc_bias
    }
}

// **** Predict ****

impl KalmanFilterXYZ {
    // Propagates the state vector forward using IMU acceleration inputs.
    /// Integrates raw IMU accelerometer data to predict new position and velocity vectors.
    ///
    /// This forms the continuous dead reckoning pipeline driving vehicle kinetics forward.
    ///
    /// ### Physical Mechanics
    /// ```math
    ///
    /// pos_k = pos_k₋₁ + vel_k₋₁ * dT + 0.5 * acc * dT²
    /// vel_k = vel_k₋₁ + acc * dT
    /// ```
    pub fn predict_state(&mut self, acc_measurement: Vector3f32, dt: f32) {
        // In NED, positive Z is down, so gravity is a positive vector
        let gravity = Vector3f32 { x: 0.0, y: 0.0, z: 9.80665 };

        // Calculate true physical acceleration by removing bias and adding gravity
        // Note: in a NED frame, gravity is a positive vector pointing downwards, so it is added to the accelerometer measurement.
        let acc = acc_measurement - self.state.acc_bias + gravity;

        // Physical mechanics
        // s = ut + 0.5 * a * t²
        self.state.pos += (self.state.vel + 0.5 * acc * dt) * dt;

        // v = u + a * t
        self.state.vel += acc * dt;

        // Bias remains constant during prediction, it is modeled as a random walk in covariance.
        // So `self.acc_bias` not updated.
    }

    /// Predict covariance:
    ///
    /// `P = F * P * Fᵀ + Q`.
    ///
    /// `P`: State covariance matrix
    /// `F`: State transition matrix
    /// `Q`: Process noise covariance.
    ///
    /// A full 9x9 matrix multiplication involves 729 individual arithmetic operations.
    /// Instead the matrix is divided into 9 separate 3x3 sub-matrices (blocks),
    /// and each block is processed separately.
    ///
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
    ///
    /// In a Kalman filter tracking Position (P), Velocity (V), and an Accelerometer Bias (B)
    /// over a small time increment (dt), the physical laws of motion give:
    ///     P_new = P + V * dt
    ///     V_new = V - B * dt  (Bias is subtracted from acceleration)
    ///     B_new = B (Bias is modeled as a constant or random walk)
    ///
    /// Expressed as a 3x3 block state transition matrix, that gives:
    ///
    ///     ┌                 9x9                  ┐
    ///     │ ┌   3x3    ┐┌   3x3    ┐┌   3x3    ┐ │
    ///     │ │    I     ││   dt*I   ││    0     │ │
    ///     │ └          ┘└          ┘└          ┘ │
    ///     │ ┌   3x3    ┐┌   3x3    ┐┌   3x3    ┐ │
    /// F = │ │    0     ││    I     ││  -dt*I   │ │
    ///     │ └          ┘└          ┘└          ┘ │
    ///     │ ┌   3x3    ┐┌   3x3    ┐┌   3x3    ┐ │
    ///     │ │    0     ││    0     ││    I     │ │
    ///     │ └          ┘└          ┘└          ┘ │
    ///     └                                      ┘
    ///
    /// Since this is a sparse matrix, we can calculate F * P * Fᵀ without doing full 9x9 matrix multiplications.
    /// Instead we perform the multiplication block by block.
    ///
    /// Expanding F * P * Fᵀ algebraically gives the following block updates:
    ///
    /// For the position columns (PP, VP, BP)
    ///     PP_new = PP + dt * (VP + PV) + dt^2 * VV
    ///     VP_new = VP + dt * VV - dt * BP - dt^2 * BV
    ///     BP_new = BP + dt * BV
    /// For the velocity columns (PV, VV, BV)
    ///     PV_new = PV + dt * VV - dt * PB - dt^2 * VB
    ///     VV_new = VV - dt * (BV + VB) + dt^2 * BB
    ///     BV_new = BV - dt * BB
    /// For the bias columns (PB, VB, BB)
    ///     PB_new = PB + dt * VB
    ///     VB_new = VB - dt * BB
    ///     BB_new = BB
    /// ```
    #[allow(unused)]
    #[allow(non_snake_case)]
    pub fn predict_covariance(&mut self, dt: f32) {
        // =====================================================================
        // PROPAGATE THE COVARIANCE P_new = (F * P_old * Fᵀ)
        // =====================================================================

        let dt2 = dt * dt;

        // Avoid taking a copy of self.P by using temporary indices and shadowing them out of scope when they have been used.
        // This ensures we cannot use an updated value in a subsequent update.
        let (PP, VP, BP, PV, VV, BV, PB, VB, BB) =
            (Self::PP, Self::VP, Self::BP, Self::PV, Self::VV, Self::BV, Self::PB, Self::VB, Self::BB);

        // --- POSITION COLUMNS (PP, VP, BP) ---
        self.P[PP] = self.P[PP] + (self.P[VP] + self.P[PV]) * dt + self.P[VV] * dt2;
        let PP = (); // Stop P_new[PP] being used in subsequent calculations.
        self.P[VP] = self.P[VP] + (self.P[VV] - self.P[BP]) * dt - self.P[BV] * dt2;
        let VP = (); // Stop P_new[VP] being used in subsequent calculations.
        self.P[BP] = self.P[BP] + self.P[BV] * dt;
        let BP = (); // Stop P_new[BP] being used in subsequent calculations.

        // --- VELOCITY COLUMNS (PV, VV, BV) ---
        self.P[PV] = self.P[PV] + (self.P[VV] - self.P[PB]) * dt - self.P[VB] * dt2;
        let PV = ();
        self.P[VV] = self.P[VV] - (self.P[BV] + self.P[VB]) * dt + self.P[BB] * dt2;
        let VV = ();
        self.P[BV] = self.P[BV] - self.P[BB] * dt;
        let BV = ();

        // --- BIAS COLUMNS (PB, VB, BB) ---
        self.P[PB] = self.P[PB] + self.P[VB] * dt;
        let PB = ();
        self.P[VB] = self.P[VB] - self.P[BB] * dt;
        // self.P[Self::BB] = self.P[Self::BB]; don't need to update BB.

        // =====================================================================
        // APPLY PROCESS NOISE (Q)
        // =====================================================================

        // Continuous process noise integrated over dt maps to the diagonal variance slots of Velocity and Bias.
        // P += Q
        self.P[Self::VV].add_diagonal_scalar_in_place(self.Q_velocity * dt);
        self.P[Self::BB].add_diagonal_scalar_in_place(self.Q_bias * dt);

        // Time propagation is highly sensitive to asymmetric shearing, so enforce symmetry on the covariance matrix.
        self.P.enforce_symmetry();
    }
}

// **** Correct ***

#[allow(non_snake_case)]
impl KalmanFilterXYZ {
    /// Phase 2 Altitude Correction using new measurement.
    /// Updates only the vertical Z axis components across all tracking states.
    ///
    /// ### Core Operations
    /// *  `S = P₂₂ + R` (Innovation Variance calculation)
    /// *  `K = P_column_2 * (1.0 / S)` (Kalman Gain column selection extraction)
    /// *  `P = P - K * (H * P)` (Covariance correction step)
    /*        let K_pos = self.P[Self::PP].column(2) * S_inv;
           let K_vel = self.P[Self::VP].column(2) * S_inv;
           let K_acc_bias = self.P[Self::BP].column(2) * S_inv;
    */
    pub fn correct_altitude(&mut self, altitude_measurement: f32, R: f32) {
        // The altitude measurement directly observes position.z.
        //
        // H = [0 0 1   0 0 0   0 0 0]

        // S = H * P * Hᵀ + R
        //
        // Since H selects position.z:
        // S = P_PP[z,z] + R

        let S = self.P[Self::PP][Matrix3x3f32::M33] + R;

        if S <= f32::EPSILON {
            return;
        }

        let S_inv = 1.0 / S;

        // P * Hᵀ
        //
        // H selects the Z position column (ie column 2), so these are:
        //
        // P_PP[:,z]
        // P_VP[:,z]
        // P_BP[:,z]

        let P_pos = self.P[Self::PP].column(2);
        let P_vel = self.P[Self::VP].column(2);
        let P_acc_bias = self.P[Self::BP].column(2);

        let K_pos = P_pos * S_inv;
        let K_vel = P_vel * S_inv;
        let K_acc_bias = P_acc_bias * S_inv;

        // State update.
        let residual = altitude_measurement - self.state.pos.z;

        self.state.pos += K_pos * residual;
        self.state.vel += K_vel * residual;
        self.state.acc_bias += K_acc_bias * residual;

        // Covariance update:
        //
        // P -= (P Hᵀ)(H P) / S
        //
        // This is a rank-1 update.
        self.P[Self::PP] -= Matrix3x3f32::outer_product(K_pos, P_pos);
        self.P[Self::PV] -= Matrix3x3f32::outer_product(K_pos, P_vel);
        self.P[Self::PB] -= Matrix3x3f32::outer_product(K_pos, P_acc_bias);

        self.P[Self::VP] -= Matrix3x3f32::outer_product(K_vel, P_pos);
        self.P[Self::VV] -= Matrix3x3f32::outer_product(K_vel, P_vel);
        self.P[Self::VB] -= Matrix3x3f32::outer_product(K_vel, P_acc_bias);

        self.P[Self::BP] -= Matrix3x3f32::outer_product(K_acc_bias, P_pos);
        self.P[Self::BV] -= Matrix3x3f32::outer_product(K_acc_bias, P_vel);
        self.P[Self::BB] -= Matrix3x3f32::outer_product(K_acc_bias, P_acc_bias);

        // Ensure numerical stability by enforcing symmetry on the covariance matrix.
        self.P.enforce_symmetry();
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
    pub fn correct_position_standard_form(
        &mut self,
        measurement: Vector3f32,
        predicted_measurement: Vector3f32,
        R: Vector3f32,
    ) {
        // S = H * P * Hᵀ + R
        // K = P * Hᵀ * S⁻¹
        // y = z - H * x
        // x += K * y
        // P -= (K * H) * P

        // Extract the submatrices representing H * P.
        let P_pos = self.P[Self::PP];
        let P_vel = self.P[Self::PV];
        let P_acc_bias = self.P[Self::PB];

        // Calculate S, the Residual Covariance matrix:
        // S = H * P * Hᵀ + R
        let S = P_pos.add_diagonal_vector(R);
        let Some(S_inv) = S.try_inverse() else {
            // The the Residual Covariance matrix may be non-invertible.
            // This happens very rarely and is due to rounding errors when the process noise covariance Q is small.
            return;
        };

        // Calculate the residual.
        // y = z - H * x
        let residual = measurement - predicted_measurement;

        // Calculate K, the Kalman gain.
        // Calculate the 3x3 segmented Kalman Gain pieces
        //
        // K = P * Hᵀ * S⁻¹
        //
        // H selects the position column block stack (Column 0: PP, VP, BP)
        // H selects the position states, so P * Hᵀ is simply the first three columns of P,
        // represented by the first three Matrix3x3 blocks.
        let K_pos = P_pos * S_inv;
        let K_vel = P_vel * S_inv;
        let K_acc_bias = P_acc_bias * S_inv;

        // Update the physical state vectors
        // x += K * y
        self.state.pos += K_pos * residual;
        self.state.vel += K_vel * residual;
        self.state.acc_bias += K_acc_bias * residual;

        // P -= K * (H * P)
        // Calculate by subtracting the K * (H * P) submatrices one by one.

        // Column 0: position column submatrices
        self.P[Self::PP] -= K_pos * P_pos;
        self.P[Self::VP] -= K_vel * P_pos;
        self.P[Self::BP] -= K_acc_bias * P_pos;

        // Column 1: velocity column submatrices
        self.P[Self::PV] -= K_pos * P_vel;
        self.P[Self::VV] -= K_vel * P_vel;
        self.P[Self::BV] -= K_acc_bias * P_vel;

        // Column 2: bias column submatrices
        self.P[Self::PB] -= K_pos * P_acc_bias;
        self.P[Self::VB] -= K_vel * P_acc_bias;
        self.P[Self::BB] -= K_acc_bias * P_acc_bias;

        // Ensure numerical stability by enforcing symmetry on the covariance matrix.
        self.P.enforce_symmetry();
    }

    /// Joseph's Stabilized Form for the covariance update step:
    ///
    /// `P{k} = (I - KH) * P{k-1} * (I - KH)ᵀ + KRKᵀ)`.
    ///
    /// While computationally more expensive, it guarantees the result remains positive-definite.
    /// That is it ensures the covariance matrix has positive eigenvalues and remains valid and invertible for future updates.
    pub fn correct_position_joseph_stabilized_form(
        &mut self,
        measurement: Vector3f32,
        predicted_measurement: Vector3f32,
        R: Vector3f32,
    ) {
        // Calculate S, the Residual Covariance matrix:
        // S = H * P * Hᵀ + R
        let S = self.P[Self::PP].add_diagonal_vector(R);

        // The the Residual Covariance matrix may be non-invertible.
        // This happens very rarely and is due to rounding errors when the process noise covariance Q is small.
        let Some(S_inv) = S.try_inverse() else {
            return;
        };

        // Calculate K, the Kalman gain.
        // `K = P * Hᵀ * S⁻¹`
        let K_pos = self.P[Self::PP] * S_inv;
        let K_vel = self.P[Self::VP] * S_inv;
        let K_acc_bias = self.P[Self::BP] * S_inv;

        // Calculate the residual.
        // y = z - H * x
        let residual = measurement - predicted_measurement;

        // Update the physical state vectors
        // x += K * y
        self.state.pos += K_pos * residual;
        self.state.vel += K_vel * residual;
        self.state.acc_bias += K_acc_bias * residual;

        // P = (I - KH) * P * (I - KH)ᵀ + KRKᵀ

        // Calculate the columns of intermediate matrix A = (I - KH) * P
        let A = Matrix3x3xM3x3::from_column_array([
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

        // Precompute transposes and helper terms
        let K_pos_t = K_pos.transpose();
        let K_vel_t = K_vel.transpose();
        let K_acc_bias_t = K_acc_bias.transpose();
        let I_minus_K_pos_t = Matrix3x3f32::identity() - K_pos_t; // Simplified distribution

        // Calculate J = A * (I - KH)ᵀ.
        // Only calculate the values that are different from A.
        let J = [
            A[Self::PP] * I_minus_K_pos_t - A[Self::PV] * K_vel_t - A[Self::PB] * K_acc_bias_t,
            A[Self::VP] * I_minus_K_pos_t - A[Self::VV] * K_vel_t - A[Self::VB] * K_acc_bias_t,
            A[Self::BP] * I_minus_K_pos_t - A[Self::BV] * K_vel_t - A[Self::BB] * K_acc_bias_t,
        ];

        // Calculate KRKᵀ blocks
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

    pub fn correct_position_xy_standard_form(&mut self, position: Vector2f32, past_pos: Vector3f32, R: Vector2f32) {
        // TODO: implement correct_position_xy_delayed.
        _ = self;
        _ = position;
        _ = R;
        _ = past_pos;
    }

    pub fn correct_position_using_anchor_distance(
        &mut self,
        anchor_position: Vector3f32,
        measured_distance: f32,
        mahalanobis_squared_threshold: f32,
        R: f32,
    ) {
        const MIN_DISTANCE: f32 = 1.0e-4;

        if measured_distance < 0.0 {
            return;
        }
        // Position relative to the anchor.
        let position = self.state.pos - anchor_position;

        // Predicted range.
        let predicted_distance = position.norm();

        // The range Jacobian is undefined at zero distance.
        if predicted_distance <= MIN_DISTANCE {
            return;
        }

        // Jacobian with respect to position:
        //
        // H = [ dx/r, dy/r, dz/r ]
        //
        // Vector3 is used as a row vector for:
        //
        //     H * P
        //
        // and as a column vector for:
        //
        //     P * Hᵀ
        let H = position / predicted_distance;

        // P * Hᵀ for each state block.
        //
        // u = P * Hᵀ
        //
        //     [ P_PP * Hᵀ ]
        // u = [ P_VP * Hᵀ ]
        //     [ P_BP * Hᵀ ]

        // P * Hᵀ
        let P_pos = self.P[Self::PP] * H;
        let P_vel = self.P[Self::VP] * H;
        let P_acc_bias = self.P[Self::BP] * H;

        // Innovation covariance:
        //
        // S = H * P * Hᵀ + R
        //   = H * P_PP * Hᵀ + R
        let S = H.dot(P_pos) + R;

        if S <= f32::EPSILON {
            return;
        }
        let S_inv = 1.0 / S;

        // Calculate the residual.
        // y = z - H * x
        let residual = measured_distance - predicted_distance;

        // Normalized residual squared (Mahalanobis distance squared).
        let mahalanobis_squared = residual * residual * S_inv;

        if mahalanobis_squared > mahalanobis_squared_threshold {
            // Reject this measurement as an outlier.
            return;
        }

        // Calculate K, the Kalman gain.
        // K = P * Hᵀ * S⁻¹
        let K_pos = P_pos * S_inv;
        let K_vel = P_vel * S_inv;
        let K_acc_bias = P_acc_bias * S_inv;

        // Update the physical state vectors
        // x += K * y
        self.state.pos += K_pos * residual;
        self.state.vel += K_vel * residual;
        self.state.acc_bias += K_acc_bias * residual;

        // Covariance update:
        //     P_new = P - K * H * P
        // where:
        //     K = P * Hᵀ / S
        // Therefore, assuming P is symmetric:
        //     P_new = P - (P * Hᵀ) * (P * Hᵀ)ᵀ / S
        // This is a rank-1 update of the full 9x9 covariance.
        self.P[Self::PP] -= Matrix3x3::outer_product(K_pos, P_pos);
        self.P[Self::PV] -= Matrix3x3::outer_product(K_pos, P_vel);
        self.P[Self::PB] -= Matrix3x3::outer_product(K_pos, P_acc_bias);

        self.P[Self::VP] -= Matrix3x3::outer_product(K_vel, P_pos);
        self.P[Self::VV] -= Matrix3x3::outer_product(K_vel, P_vel);
        self.P[Self::VB] -= Matrix3x3::outer_product(K_vel, P_acc_bias);

        self.P[Self::BP] -= Matrix3x3::outer_product(K_acc_bias, P_pos);
        self.P[Self::BV] -= Matrix3x3::outer_product(K_acc_bias, P_vel);
        self.P[Self::BB] -= Matrix3x3::outer_product(K_acc_bias, P_acc_bias);

        // Ensure numerical stability by enforcing symmetry on the covariance matrix.
        self.P.enforce_symmetry();
    }

    pub fn correct_position(&mut self, position: Vector3f32, R: Vector3f32) {
        self.correct_position_standard_form(position, self.state.pos, R);
    }

    pub fn correct_position_joseph(&mut self, position: Vector3f32, R: Vector3f32) {
        self.correct_position_joseph_stabilized_form(position, self.state.pos, R);
    }

    pub fn correct_position_xy(&mut self, position: Vector2f32, R: Vector2f32) {
        self.correct_position_xy_standard_form(position, self.state.pos, R);
    }
}

// **** Validate ***

impl KalmanFilterXYZ {
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

#[allow(unused)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KalmanStateXYZ {
    // 3D Kinematic State Vectors
    /// Position {x, y, z}.
    pub pos: Vector3f32,
    /// Velocity {x, y, z}.
    pub vel: Vector3f32,
    /// Accelerometer Bias {x, y, z}.
    pub acc_bias: Vector3f32,
}

impl Default for KalmanStateXYZ {
    fn default() -> Self {
        Self::new()
    }
}

impl KalmanStateXYZ {
    /// Constructor.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            pos: Vector3f32 { x: 0.0, y: 0.0, z: 0.0 },
            vel: Vector3f32 { x: 0.0, y: 0.0, z: 0.0 },
            acc_bias: Vector3f32 { x: 0.0, y: 0.0, z: 0.0 },
        }
    }
}

#[cfg(test)]
mod test_traits {
    use super::*;

    fn _is_normal<T: Sized + Send + Sync + Unpin>() {}
    fn is_full<T: Sized + Send + Sync + Unpin + Copy + Clone + Default + PartialEq>() {}

    #[test]
    fn normal_types() {
        is_full::<KalmanFilterXYZ>();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_3d_position_convergence() {
        let mut filter = KalmanFilterXYZ::new();

        // Capture initial uncertainty variance values from the main diagonal blocks
        let init_p_pos_x = filter.P[KalmanFilterXYZ::PP][KalmanFilterXYZ::M11];
        //let init_p_vel_x = filter.P[PositionKalmanFilter::VV][PositionKalmanFilter::M11];

        // Setup simulation pacing parameters
        let dt = 0.01; // 100 Hz tracking thread loop

        // Define realistic GPS/UWB measurement noise variances (R)
        // Tracks ±1.5m horizontal variance and ±3.0m vertical variance
        let r_gps = Vector3f32 { x: 2.25, y: 2.25, z: 9.0 };

        // Define the ground-truth physical state parameters
        let mut true_pos = Vector3f32 { x: 10.0, y: -5.0, z: 0.0 }; // Start at an offset
        let mut true_vel = Vector3f32::default();
        let true_bias = Vector3f32 { x: -0.03, y: 0.04, z: 0.01 }; // System accelerometer bias
        let gravity = Vector3f32 { x: 0.0, y: 0.0, z: 9.80665 }; // NED gravity

        // Prime the filter's initial position guess to match our starting point
        filter.state.pos = true_pos;

        println!("\n--- PHASE 1: STATIONARY GPS LOCK (1 SECOND) ---");
        #[allow(clippy::cast_possible_truncation)]
        let loops_stationary = (1.0 / dt) as i32;
        for _ in 0..loops_stationary {
            // Raw IMU reading = Kinematic Acc (0) + Bias - Gravity Reaction Force
            let acc_measurement = Vector3f32::default() + true_bias - gravity;

            filter.predict_state(acc_measurement, dt);
            filter.predict_covariance(dt);

            //filter.correct_position(true_pos, r_gps);
            filter.correct_position_joseph(true_pos, r_gps);
        }

        // --- PHASE 1 AUDIT LOGS ---
        println!("Stationary -> True Pos X: {:.4}, Estimated Pos X: {:.4}", true_pos.x, filter.pos().x);
        println!("Stationary -> True Vel X: {:.4}, Estimated Vel X: {:.4}", true_vel.x, filter.vel().x);
        println!("Stationary -> True Bias X: {:.4}, Estimated Bias X: {:.4}", true_bias.x, filter.acc_bias().x);

        // Verification assertions for Phase 1
        assert!((filter.pos().x - true_pos.x).abs() < 0.05, "Position drifted while stationary!");
        assert!((filter.vel().x - true_vel.x).abs() < 0.05, "Velocity accumulated noise while stationary!");

        // Ensure covariance has shrunk significantly below the starting bounds
        let post_stat_p_pos_x = filter.P[KalmanFilterXYZ::PP][KalmanFilterXYZ::M11];
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

            filter.predict_state(acc_measurement, dt);
            filter.predict_covariance(dt);

            filter.correct_position(true_pos, r_gps);
        }

        // --- PHASE 2 AUDIT LOGS ---
        println!("Maneuver -> True Final Pos X: {:.4}, Estimated Final Pos X: {:.4}", true_pos.x, filter.pos().x);
        println!("Maneuver -> True Final Pos Y: {:.4}, Estimated Final Pos Y: {:.4}", true_pos.y, filter.pos().y);
        println!("Maneuver -> True Final Vel X: {:.4}, Estimated Final Vel X: {:.4}", true_vel.x, filter.vel().x);
        println!("Maneuver -> True Final Vel Y: {:.4}, Estimated Final Vel Y: {:.4}", true_vel.y, filter.vel().y);

        // Tracking precision assertions
        assert!((filter.pos().x - true_pos.x).abs() < 0.1, "Filter missed true physical X position track!");
        assert!((filter.pos().y - true_pos.y).abs() < 0.1, "Filter missed true physical Y position track!");
        assert!((filter.vel().x - true_vel.x).abs() < 0.1, "Filter velocity tracking tracking failed on X axis!");
        assert!((filter.vel().y - true_vel.y).abs() < 0.1, "Filter velocity tracking tracking failed on Y axis!");

        println!("\n✅ 3D position state propagation and measurement updates verified successfully!");
    }
    #[test]
    fn test_altitude_convergence() {
        let mut filter = KalmanFilterXYZ::new();

        // Capture the initial uncertainty bounds
        let initial_p_pos = filter.P[KalmanFilterXYZ::PP][KalmanFilterXYZ::M33];
        // let initial_p_vel = filter.P[PositionKalmanFilter::VV][PositionKalmanFilter::M33];

        // Define simulation parameters
        let dt = 0.01; // 100 Hz simulation step
        let r_baro = 0.04; // 20cm variance for our barometer

        // True Physical States (Sim truths to compare against)
        let mut true_pos = Vector3f32::default();
        let mut true_vel = Vector3f32::default();

        // Explicitly track a constant real accelerometer bias in the hardware
        let true_bias = Vector3f32 { x: 0.02, y: -0.01, z: 0.05 };

        // Aerospace NED Gravity vector (Positive down)
        let gravity = Vector3f32 { x: 0.0, y: 0.0, z: 9.80665 };

        println!("\n--- PHASE 1: STATIONARY BENCH TEST (2 SECONDS) ---");
        // During this phase, an IMU resting horizontally on the ground in an NED frame
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
                    filter.P[KalmanFilterXYZ::PP][KalmanFilterXYZ::M33],
                    filter.P[KalmanFilterXYZ::VV][KalmanFilterXYZ::M33],
                    filter.P[KalmanFilterXYZ::BB][KalmanFilterXYZ::M33]
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
        let post_stationary_p_pos = filter.P[KalmanFilterXYZ::PP][KalmanFilterXYZ::M33];
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
mod test_anchors {
    use super::*;
    use num_traits::Zero;

    const EPS: f32 = 1.0e-5;

    fn v(x: f32, y: f32, z: f32) -> Vector3f32 {
        Vector3f32 { x, y, z }
    }

    fn assert_near(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() < EPS,
            "expected {}, got {}, error {}",
            expected,
            actual,
            (actual - expected).abs(),
        );
    }

    fn assert_vector_near(actual: Vector3f32, expected: Vector3f32) {
        assert_near(actual.x, expected.x);
        assert_near(actual.y, expected.y);
        assert_near(actual.z, expected.z);
    }

    fn assert_matrix_near(actual: Matrix3x3f32, expected: Matrix3x3f32) {
        for row in 0..3 {
            let a = actual.row(row);
            let e = expected.row(row);

            assert_near(a.x, e.x);
            assert_near(a.y, e.y);
            assert_near(a.z, e.z);
        }
    }

    fn make_filter() -> KalmanFilterXYZ {
        KalmanFilterXYZ {
            state: KalmanStateXYZ { pos: v(3.0, 4.0, 0.0), vel: v(0.0, 0.0, 0.0), acc_bias: v(0.0, 0.0, 0.0) },

            P: Matrix3x3xM3x3f32::identity(),

            Q_velocity: 0.0,
            Q_bias: 0.0,
        }
    }

    #[test]
    fn range_update_basic() {
        let mut kf = make_filter();

        // Position = (3,4,0), anchor = (0,0,0)
        // Predicted range = 5
        //
        // H = [0.6, 0.8, 0]
        //
        // Measurement = 6
        // residual = +1
        //
        // S = 1 + 0.01 = 1.01

        kf.correct_position_using_anchor_distance(v(0.0, 0.0, 0.0), 6.0, 100.0, 0.01);

        let scale = 1.0 / 1.01;

        assert_vector_near(kf.state.pos, v(3.0 + 0.6 * scale, 4.0 + 0.8 * scale, 0.0));

        // No cross covariance, so velocity and bias should not change.
        assert_vector_near(kf.state.vel, v(0.0, 0.0, 0.0));

        assert_vector_near(kf.state.acc_bias, v(0.0, 0.0, 0.0));
    }

    #[test]
    fn range_update_corrects_velocity_through_cross_covariance() {
        let identity = Matrix3x3f32::identity();
        let zero = Matrix3x3f32::zero();

        let cross = Matrix3x3f32::from_diagonal_element(0.5);

        let mut kf = KalmanFilterXYZ {
            state: KalmanStateXYZ { pos: v(3.0, 4.0, 0.0), vel: v(0.0, 0.0, 0.0), acc_bias: v(0.0, 0.0, 0.0) },

            P: Matrix3x3xM3x3f32::from_column_array([
                // PP, VP, BP
                identity, cross, zero, // PV, VV, BV
                cross, identity, zero, // PB, VB, BB
                zero, zero, identity,
            ]),

            Q_velocity: 0.0,
            Q_bias: 0.0,
        };

        kf.correct_position_using_anchor_distance(v(0.0, 0.0, 0.0), 6.0, 100.0, 0.01);

        // P_VP * H^T:
        //
        // [0.5 0   0]   [0.6]   [0.3]
        // [0   0.5 0] * [0.8] = [0.4]
        // [0   0   0.5] [0  ]   [0.0]
        //
        // S = 1.01
        //
        // residual = 1

        let scale = 1.0 / 1.01;

        assert_vector_near(kf.state.vel, v(0.3 * scale, 0.4 * scale, 0.0));
    }

    #[test]
    fn range_update_corrects_accelerometer_bias_through_cross_covariance() {
        let identity = Matrix3x3f32::identity();
        let zero = Matrix3x3f32::zero();

        let bp = Matrix3x3f32::from_diagonal_array([0.2, 0.3, 0.4]);
        let pb = bp;

        let mut kf = KalmanFilterXYZ {
            state: KalmanStateXYZ { pos: v(3.0, 4.0, 0.0), vel: v(0.0, 0.0, 0.0), acc_bias: v(0.0, 0.0, 0.0) },

            P: Matrix3x3xM3x3f32::from_column_array([
                // PP, VP, BP
                identity, zero, bp, // PV, VV, BV
                zero, identity, zero, // PB, VB, BB
                pb, zero, identity,
            ]),

            Q_velocity: 0.0,
            Q_bias: 0.0,
        };

        kf.correct_position_using_anchor_distance(v(0.0, 0.0, 0.0), 6.0, 100.0, 0.01);

        // P_BP * H^T =
        //
        // [0.2 * 0.6] = 0.12
        // [0.3 * 0.8] = 0.24
        // [0.4 * 0.0] = 0.00

        let scale = 1.0 / 1.01;

        assert_vector_near(kf.state.acc_bias, v(0.12 * scale, 0.24 * scale, 0.0));
    }

    #[test]
    fn perfect_range_does_not_change_state() {
        let mut kf = make_filter();

        let original_pos = kf.state.pos;
        let original_vel = kf.state.vel;
        let original_bias = kf.state.acc_bias;

        // Predicted range is exactly 5.
        kf.correct_position_using_anchor_distance(v(0.0, 0.0, 0.0), 5.0, 100.0, 0.01);

        assert_vector_near(kf.state.pos, original_pos);
        assert_vector_near(kf.state.vel, original_vel);
        assert_vector_near(kf.state.acc_bias, original_bias);
    }

    #[test]
    fn outlier_is_rejected_without_changing_filter() {
        let mut kf = make_filter();

        let original_pos = kf.state.pos;
        let original_vel = kf.state.vel;
        let original_bias = kf.state.acc_bias;
        let original_p = kf.P;

        // Predicted range = 5.
        // 100 m is obviously an outlier.
        kf.correct_position_using_anchor_distance(v(0.0, 0.0, 0.0), 100.0, 6.63, 0.01);

        assert_vector_near(kf.state.pos, original_pos);
        assert_vector_near(kf.state.vel, original_vel);
        assert_vector_near(kf.state.acc_bias, original_bias);

        assert_matrix_near(kf.P[KalmanFilterXYZ::PP], original_p[KalmanFilterXYZ::PP]);
        assert_matrix_near(kf.P[KalmanFilterXYZ::PV], original_p[KalmanFilterXYZ::PV]);
        assert_matrix_near(kf.P[KalmanFilterXYZ::PB], original_p[KalmanFilterXYZ::PB]);
        assert_matrix_near(kf.P[KalmanFilterXYZ::VP], original_p[KalmanFilterXYZ::VP]);
        assert_matrix_near(kf.P[KalmanFilterXYZ::VV], original_p[KalmanFilterXYZ::VV]);
        assert_matrix_near(kf.P[KalmanFilterXYZ::VB], original_p[KalmanFilterXYZ::VB]);
        assert_matrix_near(kf.P[KalmanFilterXYZ::BP], original_p[KalmanFilterXYZ::BP]);
        assert_matrix_near(kf.P[KalmanFilterXYZ::BV], original_p[KalmanFilterXYZ::BV]);
        assert_matrix_near(kf.P[KalmanFilterXYZ::BB], original_p[KalmanFilterXYZ::BB]);
    }
    #[test]
    fn range_update_reduces_radial_position_variance() {
        let mut kf = make_filter();

        let h = v(0.6, 0.8, 0.0);

        let before = h.dot(kf.P[KalmanFilterXYZ::PP] * h);

        kf.correct_position_using_anchor_distance(v(0.0, 0.0, 0.0), 5.0, 100.0, 0.01);

        let after = h.dot(kf.P[KalmanFilterXYZ::PP] * h);

        assert!(after < before, "radial variance should decrease: before={before}, after={after}");

        // With identity P and a range-only measurement,
        // the direction perpendicular to H should retain its
        // original variance.
        let perpendicular = v(-0.8, 0.6, 0.0);

        let before_perp = perpendicular.dot(kf.P[KalmanFilterXYZ::PP] * perpendicular);

        assert_near(before_perp, 1.0);

        // Z is completely unobserved by this range measurement.
        let z = v(0.0, 0.0, 1.0);

        let z_variance = z.dot(kf.P[KalmanFilterXYZ::PP] * z);

        assert_near(z_variance, 1.0);
    }
}
