use vqm::{Matrix3x3f32, Matrix9f32, Vector3f32};

use crate::KalmanStateVector9f32;

/// `f32` variant of `PositionKalmanFilter0`.
pub type PositionKalmanFilter9f32 = PositionKalmanFilter9;

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
pub struct PositionKalmanFilter9 {
    // 3D Kinematic State Vectors
    /// Position (x, y, z).
    pub pos: Vector3f32,
    /// Velocity (x, y, z).
    pub vel: Vector3f32,
    /// Accelerometer Bias (x, y, z).
    pub acc_bias: Vector3f32,

    /// Predicted System Uncertainty Covariance Matrix (P).
    pub P: Matrix9f32,
    /// Estimated Post-Correction Error Covariance Matrix (E).
    pub E: Matrix9f32,

    // --- Hyperparameters & Tuning Constants ---
    /// Process Noise spectral density mapping to Velocity variance.
    pub q_velocity: f32,
    /// Process Noise spectral density mapping to Sensor Drift variance.
    pub q_bias: f32,
    /// Absolute Measurement Noise variance for horizontal GPS channels.
    pub r_gps_horizontal: f32,
    /// Absolute Measurement Noise variance for vertical GPS channels.
    pub r_gps_vertical: f32,
    /// Absolute Measurement Noise variance for barometric pressure altimeter.
    pub r_barometer: f32,
    /// Absolute Measurement Noise variance for rangefinder.
    pub r_rangefinder: f32,
    /// Absolute Measurement Noise variance for optical flow.
    pub r_optical_flow: Vector3f32,
}

impl Default for PositionKalmanFilter9 {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(missing_docs)]
impl PositionKalmanFilter9 {
    pub const Z_POS_ROW: usize = 2; // H vector selects the 3rd row of P
    pub const Z_POS_COL: usize = 2; // 3rd column corresponds to Z position (Altitude)
    pub const S_XX: usize = Matrix3x3f32::M11;
    pub const S_YY: usize = Matrix3x3f32::M22;
    pub const S_ZZ: usize = Matrix3x3f32::M33;
}

impl PositionKalmanFilter9 {
    /// Constructor.
    #[must_use]
    pub fn new() -> Self {
        Self {
            pos: Vector3f32 { x: 0.0, y: 0.0, z: 0.0 },
            vel: Vector3f32 { x: 0.0, y: 0.0, z: 0.0 },
            acc_bias: Vector3f32 { x: 0.0, y: 0.0, z: 0.0 },
            q_velocity: 0.0,
            q_bias: 0.0,
            r_gps_horizontal: 0.0,
            r_gps_vertical: 0.0,
            r_barometer: 0.0,
            r_rangefinder: 0.0,
            r_optical_flow: Vector3f32 { x: 0.0, y: 0.0, z: 0.0 },
            E: Matrix9f32::default(),
            P: Matrix9f32::default(),
        }
    }
}

// **** Predict ****

impl PositionKalmanFilter9 {
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
    /*
        So I have a Kalman filter implementation the uses Matrix9x9, a 9x9 matrix of 81 elements stored in a flat array in column-major order.
        The predict function split the 9x9 matrix into 9 3x3 blocks as below.
        I'd like to reimplement it with the new Matrix9, we are using, ie 9 Matrix3x3 stored in a flat array in column-major order.
        Can you help me with that?
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
    pub fn predict_covariance(&mut self, dt: f32) {
        // -------------------------------------------------------------------------
        // E is a 3x3 matrix of 3x3 blocks:
        //
        //                  Position   Velocity   Bias
        //               ┌──────────┬──────────┬──────────┐
        //      Position │ E[PP]    │ E[PV]    │ E[PB]    │
        //               ├──────────┼──────────┼──────────┤
        //      Velocity │ E[VP]    │ E[VV]    │ E[VB]    │
        //               ├──────────┼──────────┼──────────┤
        //          Bias │ E[BP]    │ E[BV]    │ E[BB]    │
        //               └──────────┴──────────┴──────────┘
        //
        // Matrix9 stores these column-major:
        //
        //     a[0] a[3] a[6]
        //     a[1] a[4] a[7]
        //     a[2] a[5] a[8]
        // -------------------------------------------------------------------------

        const S_XX: usize = Matrix3x3f32::M11;
        const S_YY: usize = Matrix3x3f32::M22;
        const S_ZZ: usize = Matrix3x3f32::M33;

        const PP: usize = 0;
        const VP: usize = 1;
        const BP: usize = 2;

        const PV: usize = 3;
        const VV: usize = 4;
        const BV: usize = 5;

        /*const PB: usize = 6;
        const VB: usize = 7;*/
        const BB: usize = 8;

        let one_plus_dt = 1.0 + dt;
        let one_minus_dt = 1.0 - dt;

        let E = self.E;
        let mut P = E;

        // =====================================================================
        // POSITION COLUMNS
        // =====================================================================

        // Position / Position
        P[PP] = E[PP] + dt * (E[VP] * one_plus_dt + E[PV]);

        // Velocity / Position
        P[VP] = E[PV] - dt * (E[VP] * one_plus_dt - E[BP]);

        // Bias / Position
        P[BP] = E[BP] + dt * E[VP];

        // =====================================================================
        // VELOCITY COLUMNS
        // =====================================================================

        // Position / Velocity
        P[PV] = E[VP] + dt * (E[BP] * one_minus_dt - E[VV]);

        // Velocity / Velocity
        P[VV] = E[VV] - dt * (E[BP] * one_plus_dt + E[BV]);

        // Bias / Velocity
        P[BV] = E[BV];

        // =====================================================================
        // BIAS COLUMNS
        //
        // PB, VB and BB are unchanged, so they are already correct because
        // P was initialized from E above.
        // =====================================================================

        // =====================================================================
        // PROCESS NOISE
        // =====================================================================

        let q_velocity_dt2 = self.q_velocity * dt * dt;

        P[VV][S_XX] += q_velocity_dt2;
        P[VV][S_YY] += q_velocity_dt2;
        P[VV][S_ZZ] += q_velocity_dt2;

        let q_bias_dt2 = self.q_bias * dt * dt;

        P[BB][S_XX] += q_bias_dt2;
        P[BB][S_YY] += q_bias_dt2;
        P[BB][S_ZZ] += q_bias_dt2;

        self.P = P;
    }
}

// **** Correct ***

impl PositionKalmanFilter9 {
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
        let S = self.P[0][Matrix9f32::M33] + R;

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
        self.E = self.P - K.outer_product9(altitude_row);
        self.P = self.E;
    }

    /// Phase 2: Correct altitude using the barometer measurement.
    #[inline]
    pub fn correct_altitude_using_barometer(&mut self, altitude: f32) {
        self.correct_altitude(altitude, self.r_barometer);
    }

    /// Phase 2: Correct altitude using the rangefinder measurement.
    #[inline]
    pub fn correct_altitude_using_rangefinder(&mut self, altitude: f32) {
        self.correct_altitude(altitude, self.r_rangefinder);
    }

    /// Phase 2: Correct altitude using GPS vertical measurement.
    #[inline]
    pub fn correct_altitude_using_gps(&mut self, altitude: f32) {
        self.correct_altitude(altitude, self.r_gps_vertical);
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
        //let (K_pos, K_vel, K_acc_bias) = Matrix9x9f32::multiply_9x3_by_3x3(&self.P, S_inv);
        let K_pos = self.P[0] * S_inv;
        let K_vel = self.P[0] * S_inv;
        let K_acc_bias = self.P[0] * S_inv;

        // Calculate the error vector.
        let error = position - self.pos;

        // Update the state vectors across all three physical domains.
        self.pos += K_pos * error;
        self.vel += K_vel * error;
        self.acc_bias += K_acc_bias * error;

        // Calculate K * (H * P) by re-assembling the 3x3 K_matrices into the 9x9 KH_P matrix.
        let KH_P = Matrix9f32::identity(); //self.reassemble_k_matrices(K_pos, K_vel, K_acc_bias);

        // Update Covariance Matrix: E = P - K * (H * P)
        self.E = self.P - KH_P;

        // Synchronize the active covariance state for the next prediction phase
        self.P = self.E;
    }

    /// Phase 2: Correct position using GPS position measurement (typically at a 1Hz to 10Hz rate).
    pub fn correct_position_using_gps(&mut self, position: Vector3f32) {
        let r_gps = Vector3f32 { x: self.r_gps_horizontal, y: self.r_gps_horizontal, z: self.r_gps_vertical };
        self.correct_position(position, r_gps);
    }

    /// Phase 2: Correct position using optical flow position measurement.
    pub fn correct_position_using_optical_flow(&mut self, position: Vector3f32) {
        self.correct_position(position, self.r_optical_flow);
    }
}

// **** Validate ***

impl PositionKalmanFilter9 {
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
    pub fn validate_measurement(&self, y: Vector3f32, _R: Vector3f32, gate_threshold: f32) -> bool {
        // Collect the columns into an array using standard iteration.
        // Collecting exactly 3 items ensures we can pattern match them safely without using `unwrap`.
        let mut col_iter = self.P.iter_columns();
        let (Some(col0), Some(col1), Some(col2)) = (col_iter.next(), col_iter.next(), col_iter.next()) else {
            return false; // Structured pipeline fallback safety
        };

        // H selects position states (rows 0, 1, 2) from columns 0, 1, 2 of matrix P.
        // We pack these into our a Matrix3x3f32.
        /*#[rustfmt::skip]
        let S = Matrix3x3f32::from_column_array([
            col0[0] + R.x, col0[1],       col0[2],
            col1[0],       col1[1] + R.y, col1[2],
            col2[0],       col2[1],       col2[2] + R.z,
        ]);*/
        _ = col0;
        _ = col1;
        _ = col2;

        let S = Matrix3x3f32::identity();

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
        is_full::<PositionKalmanFilter9>();
    }
}

#[cfg(test)]
mod tests {
    use vqm::Matrix9x9f32;

    use crate::PositionKalmanFilter;

    use super::*;

    fn assert_matrix9x9_close(expected: &Matrix9x9f32, actual: &Matrix9x9f32, epsilon: f32) {
        for col in 0..9 {
            for row in 0..9 {
                let index = col * 9 + row;

                let expected_value = expected[index];
                let actual_value = actual[index];
                let difference = (expected_value - actual_value).abs();

                assert!(
                    difference <= epsilon,
                    "Matrix mismatch at ({row}, {col}), index {index}: \
                     expected {expected_value}, got {actual_value}, \
                     difference {difference}"
                );
            }
        }
    }

    #[test]
    fn predict_covariance_block_implementation_matches_original() {
        // Deliberately non-symmetric matrix. This is useful because it
        // exposes row/column and transpose errors that could be hidden
        // by a symmetric covariance matrix.
        let initial_e = Matrix9x9f32::from_row_array([
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0,
            20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0,
            38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0,
            56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0,
            74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0, 81.0,
        ]);

        let dt = 0.01;
        let q_velocity = 0.2;
        let q_bias = 0.03;

        let mut original = PositionKalmanFilter::new();
        original.E = initial_e;
        original.q_velocity = q_velocity;
        original.q_bias = q_bias;

        // Run the original scalar implementation.
        original.predict_covariance(dt);

        let mut block = PositionKalmanFilter9::new();
        block.E = Matrix9f32::from(initial_e);
        block.q_velocity = q_velocity;
        block.q_bias = q_bias;

        // Run the new Matrix9 block implementation.
        block.predict_covariance(dt);

        // The two implementations should produce the same 9x9 matrix.
        assert_matrix9x9_close(&original.P, &Matrix9x9f32::from(block.P), 1e-5);
    }
}
