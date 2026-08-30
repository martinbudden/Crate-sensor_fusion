use vqm::{Matrix3x3f32, Matrix9x9f32, Vector3f32};

use crate::KalmanStateVector9f32;

/// `f32` variant of `PositionKalmanFilter`.
pub type PositionKalmanFilterf32 = PositionKalmanFilter;

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
    pub P: Matrix9x9f32,
    /// Estimated Post-Correction Error Covariance Matrix (E).
    pub E: Matrix9x9f32,

    // --- Hyperparameters & Tuning Constants ---
    /// Process Noise spectral density mapping to Velocity variance.
    pub q_velocity: f32,
    /// Process Noise spectral density mapping to Sensor Drift variance.
    pub q_bias: f32,
}

impl Default for PositionKalmanFilter {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(missing_docs)]
impl PositionKalmanFilter {
    pub const Z_POS_ROW: usize = 2; // H vector selects the 3rd row of P
    pub const Z_POS_COL: usize = 2; // 3rd column corresponds to Z position (Altitude)
    pub const S_XX: usize = Matrix3x3f32::M11;
    pub const S_YY: usize = Matrix3x3f32::M22;
    pub const S_ZZ: usize = Matrix3x3f32::M33;
}

impl PositionKalmanFilter {
    /// Constructor.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            pos: Vector3f32 { x: 0.0, y: 0.0, z: 0.0 },
            vel: Vector3f32 { x: 0.0, y: 0.0, z: 0.0 },
            acc_bias: Vector3f32 { x: 0.0, y: 0.0, z: 0.0 },
            q_velocity: 0.0,
            q_bias: 0.0,
            E: Matrix9x9f32::new([0.0; 81]),
            P: Matrix9x9f32::new([0.0; 81]),
        }
    }
}

// **** Predict ****

impl PositionKalmanFilter {
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
    pub fn predict_states(&mut self, acc_measurement: Vector3f32, dt: f32) {
        let gravity = Vector3f32 { x: 0.0, y: 0.0, z: 9.80665 };

        // Calculate true physical acceleration by removing bias and adding gravity
        let acc_true = acc_measurement - self.acc_bias - gravity;

        // High-level vector physics integration
        self.pos += (self.vel + 0.5 * acc_true * dt) * dt;
        self.vel += acc_true * dt;
        // Bias remains constant during prediction, it is modeled as a random walk in covariance.
    }
    /*
    Our state vector is organized as [{p}, {v}, {b}]^T.
    The kinematic transition equations using simple Euler integration are:
    {p}_k = {p}_{k-1} + {v}_{k-1}Delta T
    {v}_k = {v}_{k-1} - vec{b}_{k-1}Delta T (assuming acceleration is updated via the control loop)
    {b}_k = {b}_{k-1}
    This means our 9x9 matrix **A** is incredibly sparse, containing only a few *dt* terms on the off-diagonals.
     If we write out the math for \(AEA^{T}\) manually using 3x3 blocks, the matrix operations simplify into a clean sequence of 3x3 array updates.
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
    /// ```
    /// ## Formula
    /// *  `P_k = A * E_k₋₁ * Aᵀ + Q`
    #[allow(non_snake_case)]
    #[rustfmt::skip]
    pub fn predict_covariance(&mut self, dt: f32) {
        // Calculate
        //    P = A * E * A^T + Q
        // avoiding expensive 9x9 matrix multiplication by splitting the 9x9 matrices into sub-matrices (blocks).
        let mut P = Matrix9x9f32::default();

        // Instantiate the custom column iterator
        // so `col_iter` points to the entire matrix: [Col0, Col1, Col2, Col3, Col4, Col5, Col6, Col7, Col8]
        let mut col_iter = P.iter_columns_mut();

        let one_plus_dt = 1.0 + dt;
        let one_minus_dt = 1.0 - dt;

        // =====================================================================
        // SECTION 1: POSITION COLUMNS (COLUMNS 0..=2 ( - c ranges 0..3)
        // =====================================================================
        // .by_ref() lets us consume 3 items without relinquishing ownership of the iterator
        // so section 2 will continue where section 1 left off.
        for (c, P_col) in col_iter.by_ref().take(3).enumerate() {
            // Loop Iteration 1: c = 0, P_col = &mut Col0
            // Loop Iteration 2: c = 1, P_col = &mut Col1
            // Loop Iteration 3: c = 2, P_col = &mut Col2

            // PositionPosition Block (Rows 0..3)
            P_col[0] = self.E[c]      + dt * (self.E[c + 3]  * one_plus_dt + self.E[c + 27]);
            P_col[1] = self.E[c + 9]  + dt * (self.E[c + 12] * one_plus_dt + self.E[c + 36]);
            P_col[2] = self.E[c + 18] + dt * (self.E[c + 21] * one_plus_dt + self.E[c + 45]);

            // VelocityPosition Block (Rows 3..6)
            P_col[3] = self.E[c + 27] - dt * (self.E[c + 3]  * one_plus_dt - self.E[c + 54]);
            P_col[4] = self.E[c + 36] - dt * (self.E[c + 12] * one_plus_dt - self.E[c + 63]);
            P_col[5] = self.E[c + 45] - dt * (self.E[c + 21] * one_plus_dt - self.E[c + 72]);

            // BiasPosition Block (Rows 6..9)
            P_col[6] = self.E[c + 54] + dt * self.E[c + 3];
            P_col[7] = self.E[c + 63] + dt * self.E[c + 12];
            P_col[8] = self.E[c + 72] + dt * self.E[c + 21];
        }
        // =====================================================================
        // SECTION 2: COLUMNS 3..=5 (VELOCITY COLUMNS)
        // =====================================================================
        for (c, P_col) in col_iter.by_ref().take(3).enumerate() {
            // Loop Iteration 1: c = 0, P_col = &mut Col3
            // Loop Iteration 2: c = 1, P_col = &mut Col4
            // Loop Iteration 3: c = 2, P_col = &mut Col5

            // PositionVelocity Block (Rows 0..3)
            P_col[0] = self.E[c + 3]  + dt * (self.E[c + 6]  * one_minus_dt - self.E[c + 30]);
            P_col[1] = self.E[c + 12] + dt * (self.E[c + 15] * one_minus_dt - self.E[c + 39]);
            P_col[2] = self.E[c + 21] + dt * (self.E[c + 24] * one_minus_dt - self.E[c + 48]);

            // VelocityVelocity Block (Rows 3..6)
            P_col[3] = self.E[c + 30] - dt * (self.E[c + 6]  * one_plus_dt + self.E[c + 57]);
            P_col[4] = self.E[c + 39] - dt * (self.E[c + 15] * one_plus_dt + self.E[c + 66]);
            P_col[5] = self.E[c + 48] - dt * (self.E[c + 24] * one_plus_dt + self.E[c + 75]);

            // BiasVelocity Block (Rows 6..9)
            P_col[6] = self.E[c + 57];
            P_col[7] = self.E[c + 66];
            P_col[8] = self.E[c + 75];
        }
        // =====================================================================
        // SECTION 3: COLUMNS 6..=8 (BIAS COLUMNS)
        // =====================================================================
        // Don't need .by_ref() or take(3), since we are consuming the remainder 3 items of the iterator.
        for (c, P_col) in col_iter.enumerate() {
            // Loop Iteration 1: c = 0, P_col = &mut Col6
            // Loop Iteration 2: c = 1, P_col = &mut Col7
            // Loop Iteration 3: c = 2, P_col = &mut Col8
            P_col[0] = self.E[c + 6];
            P_col[1] = self.E[c + 15];
            P_col[2] = self.E[c + 24];
            P_col[3] = self.E[c + 33];
            P_col[4] = self.E[c + 42];
            P_col[5] = self.E[c + 51];
            P_col[6] = self.E[c + 60];
            P_col[7] = self.E[c + 69];
            P_col[8] = self.E[c + 78];
        }
        // =====================================================================
        // PROCESS NOISE INJECTION (Additive Q terms on the active diagonals)
        // =====================================================================

        // Add Q.

        // Velocity random walk noise maps to diagonal states 4, 5, and 6
        let q_velocity_dt2 = self.q_velocity * dt * dt;
        P[Matrix9x9f32::M44] += q_velocity_dt2;
        P[Matrix9x9f32::M55] += q_velocity_dt2;
        P[Matrix9x9f32::M66] += q_velocity_dt2;

        // Accelerometer bias random walk noise maps to diagonal states 7, 8, and 9
        let q_bias_dt2 = self.q_bias * dt * dt;
        P[Matrix9x9f32::M77] += q_bias_dt2;
        P[Matrix9x9f32::M88] += q_bias_dt2;
        P[Matrix9x9f32::M99] += q_bias_dt2;

        self.P = P;
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
    #[allow(non_snake_case)]
    pub fn correct_altitude(&mut self, altitude: f32, R: f32) {
        // Calculate the scalar innovation covariance: S = P_zz + R
        let S = self.P[Matrix9x9f32::M33] + R;

        // Calculate the 9-element Kalman Gain vector: K = (P * H^T) / S
        // Multiplying P by H^T is mathematically identical to extracting the 3rd column of P
        let K = KalmanStateVector9f32::from(self.P.column_tuple_vector(Self::Z_POS_COL)) * (1.0 / S);

        // Calculate the scalar innovation error
        let error = altitude - self.pos.z;

        // Update the state vectors
        self.pos += K.pos * error;
        self.vel += K.vel * error;
        self.acc_bias += K.bias * error;

        // Extract the altitude row of P to calculate the error covariance: E = P - K * H * P
        let altitude_row = KalmanStateVector9f32::from(self.P.row_tuple_vector(Self::Z_POS_ROW));

        // K.outer_product(altitude_row) generates the 9x9 correction matrix
        self.E = self.P - K.outer_product(altitude_row);
        self.P = self.E;
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
    /// * position: Vector3f32 observation `[z_x, z_y, z_z]`
    /// * R: Vector3f32 diagonal measurement noise variance [R.x, R.y, R.z]
    #[allow(non_snake_case)]
    pub fn correct_position(&mut self, position: Vector3f32, R: Vector3f32) {
        // Extract the PositionPosition 3x3 sub-matrix from the top-left of the 9x9 P matrix.
        let mut P_pos = Matrix3x3f32::from(self.P);

        // Calculate the 3x3 Innovation Covariance matrix: S = H * P * H^T + R
        // In our model, R is a diagonal matrix containing horizontal and vertical sensory noise.
        P_pos[Self::S_XX] += R.x;
        P_pos[Self::S_YY] += R.y;
        P_pos[Self::S_ZZ] += R.z;

        // Calculate inverse of S.
        // If S is singular (eg sensor fault), we safely return to prevent a system crash.
        let Some(S_inv) = P_pos.try_inverse() else {
            return;
        };

        // Calculate the Kalman Gain: K = (P * H^T) * S_inv, and split it into 3 separate 3x3 matrices.
        // We do this by extracting the first 3 columns of P, which is mathematically equivalent to calculating P * H^T
        // and then multiplying by S_inv.
        let (K_pos, K_vel, K_acc_bias) = Matrix9x9f32::multiply_9x3_by_3x3(&self.P, S_inv);

        // Calculate the error vector.
        let error = position - self.pos;

        // Update the state vectors across all three physical domains.
        self.pos += K_pos * error;
        self.vel += K_vel * error;
        self.acc_bias += K_acc_bias * error;

        // Calculate K * (H * P) by re-assembling the 3x3 K_matrices into the 9x9 KH_P matrix.
        let KH_P = self.reassemble_k_matrices(K_pos, K_vel, K_acc_bias);

        // Update Covariance Matrix: E = P - K * (H * P)
        self.E = self.P - KH_P;

        // Synchronize the active covariance state for the next prediction phase
        self.P = self.E;
    }

    /// Reassembles the `K_pos`, `K_vel`, and `K_acc_bias` 3x3 matrices into a 9x9 matrix
    /// using an optimal column-major layout pipeline.
    #[allow(non_snake_case)]
    #[rustfmt::skip]
    #[must_use]
    pub fn reassemble_k_matrices(
        &self,
        K_pos: Matrix3x3f32,
        K_vel: Matrix3x3f32,
        K_acc_bias: Matrix3x3f32,
    ) -> Matrix9x9f32 {
        let mut KH_P = Matrix9x9f32::default();

        // Cache all the K matrix columns in registers.
        let mut kp_iter = K_pos.iter_columns();
        let (Some(kp0), Some(kp1), Some(kp2)) = (kp_iter.next(), kp_iter.next(), kp_iter.next()) else {
            return KH_P;
        };

        let mut kv_iter = K_vel.iter_columns();
        let (Some(kv0), Some(kv1), Some(kv2)) = (kv_iter.next(), kv_iter.next(), kv_iter.next()) else {
            return KH_P;
        };

        let mut kb_iter = K_acc_bias.iter_columns();
        let (Some(kb0), Some(kb1), Some(kb2)) = (kb_iter.next(), kb_iter.next(), kb_iter.next()) else {
            return KH_P;
        };

        // Stream column by column through the 9x9 matrices.
        // Zip the read-only columns of self.P with the mutable output columns of KH_P.
        let p_cols = self.P.iter_columns();
        let mut out_cols = KH_P.iter_columns_mut();

        for (P_col, out_col) in p_cols.zip(&mut out_cols) {
            // H selects rows 0, 1, 2 of the current P column.
            // These serve as our scalar weights for the linear combination.
            let p0 = P_col[0];
            let p1 = P_col[1];
            let p2 = P_col[2];

            // --- Rows 0..3: Position States (K_pos * v) ---
            out_col[0] = p0 * kp0[0] + p1 * kp1[0] + p2 * kp2[0];
            out_col[1] = p0 * kp0[1] + p1 * kp1[1] + p2 * kp2[1];
            out_col[2] = p0 * kp0[2] + p1 * kp1[2] + p2 * kp2[2];

            // --- Rows 3..6: Velocity States (K_vel * v) ---
            out_col[3] = p0 * kv0[0] + p1 * kv1[0] + p2 * kv2[0];
            out_col[4] = p0 * kv0[1] + p1 * kv1[1] + p2 * kv2[1];
            out_col[5] = p0 * kv0[2] + p1 * kv1[2] + p2 * kv2[2];

            // --- Rows 6..9: Accelerometer Bias States (K_acc_bias * v) ---
            out_col[6] = p0 * kb0[0] + p1 * kb1[0] + p2 * kb2[0];
            out_col[7] = p0 * kb0[1] + p1 * kb1[1] + p2 * kb2[1];
            out_col[8] = p0 * kb0[2] + p1 * kb1[2] + p2 * kb2[2];
        }

        KH_P
    }
}

impl PositionKalmanFilter {
    /// Performs a numerically stable Joseph Form Covariance Update.
    /// Formula: `P_new = (I - KH) * P * (I - KH)ᵀ + K * R * Kᵀ`.
    ///
    /// Because `H` is a sparse observation matrix selecting only the first 3 rows (position states),
    ///  we can break the formula down into two efficient operations:
    /// The `(I - KH) * P` Term: Expressed as`P - KHP` (this subtraction operates perfectly down contiguous column lines).
    /// The Joseph Term  `K * R * Kᵀ`: since `R` is diagonal this simplifies down to a weighted outer product of the columns of `K`.
    ///
    /// Layouts:
    /// * self.P: 9x9 Column-Major Covariance Matrix
    /// * `KH_P`: 9x9 Column-Major Matrix from `reassemble_k_matrices`
    /// * `K_pos`, `K_vel`, `K_acc_bias`: 3x3 Column-Major Kalman Gain blocks
    /// * R: Vector3f32 diagonal measurement noise variance [R.x, R.y, R.z]
    #[allow(non_snake_case)]
    #[rustfmt::skip]
    pub fn joseph_covariance_update(
        &mut self,
        KH_P: &Matrix9x9f32,
        K_pos: &Matrix3x3f32,
        K_vel: &Matrix3x3f32,
        K_acc_bias: &Matrix3x3f32,
        R: Vector3f32,
    ) {
        // =====================================================================
        // Cache all Kalman gain columns for the KRKᵀ outer product
        // =====================================================================
        let mut kp_iter = K_pos.iter_columns();
        let (Some(kp0), Some(kp1), Some(kp2)) = (kp_iter.next(), kp_iter.next(), kp_iter.next()) else { return; };

        let mut kv_iter = K_vel.iter_columns();
        let (Some(kv0), Some(kv1), Some(kv2)) = (kv_iter.next(), kv_iter.next(), kv_iter.next()) else { return; };

        let mut kb_iter = K_acc_bias.iter_columns();
        let (Some(kb0), Some(kb1), Some(kb2)) = (kb_iter.next(), kb_iter.next(), kb_iter.next()) else { return; };

        // Construct the full 9x3 Kalman Gain matrix columns directly into registers
        let K_col0 = [kp0[0], kp0[1], kp0[2], kv0[0], kv0[1], kv0[2], kb0[0], kb0[1], kb0[2]];
        let K_col1 = [kp1[0], kp1[1], kp1[2], kv1[0], kv1[1], kv1[2], kb1[0], kb1[1], kb1[2]];
        let K_col2 = [kp2[0], kp2[1], kp2[2], kv2[0], kv2[1], kv2[2], kb2[0], kb2[1], kb2[2]];

        // =====================================================================
        // Calculate the first order linear term: M = (I - KH)P = P - KH_P
        // =====================================================================
        let mut M = Matrix9x9f32::default();

        let P_cols = self.P.iter_columns();
        let KHP_cols = KH_P.iter_columns();
        let mut M_cols = M.iter_columns_mut();

        for ((P_col, KH_P_col), M_col) in P_cols.zip(KHP_cols).zip(&mut M_cols) {
            // Update the data inside the column slice row-by-row
            M_col[0] = P_col[0] - KH_P_col[0];
            M_col[1] = P_col[1] - KH_P_col[1];
            M_col[2] = P_col[2] - KH_P_col[2];
            M_col[3] = P_col[3] - KH_P_col[3];
            M_col[4] = P_col[4] - KH_P_col[4];
            M_col[5] = P_col[5] - KH_P_col[5];
            M_col[6] = P_col[6] - KH_P_col[6];
            M_col[7] = P_col[7] - KH_P_col[7];
            M_col[8] = P_col[8] - KH_P_col[8];
        }

        // =====================================================================
        // Complete the Joseph form correction stride
        // =====================================================================
        let mut M_col_iter = M.iter_columns();
        let (Some(m0), Some(m1), Some(m2)) = (M_col_iter.next(), M_col_iter.next(), M_col_iter.next()) else {
            return;
        };

        let M_cols_read = M.iter_columns();
        let KHP_cols_read = KH_P.iter_columns();
        let mut P_cols_write = self.P.iter_columns_mut();

        for ((M_col, KH_P_col), P_col) in M_cols_read.zip(KHP_cols_read).zip(&mut P_cols_write) {
            let kh0 = KH_P_col[0];
            let kh1 = KH_P_col[1];
            let kh2 = KH_P_col[2];

            // Run the unrolled matrix update loop across all 9 internal rows
            for r in 0..9 {
                // Calculate the core M * (I - KH)ᵀ element value
                let mut p_updated = M_col[r] - (m0[r] * kh0 + m1[r] * kh1 + m2[r] * kh2);

                // Add the additive noise vector components using explicit field names from Vector3f32
                p_updated += K_col0[r] * R.x * K_col0[r]
                           + K_col1[r] * R.y * K_col1[r]
                           + K_col2[r] * R.z * K_col2[r];

                P_col[r] = p_updated;
            }
        }
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
        let mut col_iter = self.P.iter_columns();
        let (Some(col0), Some(col1), Some(col2)) = (col_iter.next(), col_iter.next(), col_iter.next()) else {
            return false; // Structured pipeline fallback safety
        };

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

#[cfg(test)]
mod test_traits {
    use super::*;

    fn _is_normal<T: Sized + Send + Sync + Unpin>() {}
    fn is_full<T: Sized + Send + Sync + Unpin + Copy + Clone + Default + PartialEq>() {}

    #[test]
    fn normal_types() {
        is_full::<PositionKalmanFilter>();
    }
}
