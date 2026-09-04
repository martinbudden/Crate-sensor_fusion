use vqm::{Matrix2x2f32, Matrix3x3xM2x2, Matrix3x3xM2x2f32, Vector2f32};

/// `f32` variant of `PositionKalmanFilter0`.
pub type KalmanFilterXYf32 = KalmanFilterXY;

#[allow(non_snake_case)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KalmanFilterXY {
    // 2D Kinematic State Vectors
    /// Position (x, y).
    pub pos: Vector2f32,
    /// Velocity (x, y).
    pub vel: Vector2f32,
    /// Accelerometer Bias (x, y).
    pub acc_bias: Vector2f32,

    /// Predicted System Uncertainty Covariance Matrix (P).
    /// **P*: Prediction error covariance (the system's internal uncertainty).
    pub P: Matrix3x3xM2x2f32,

    // state transition noise covariance Matrix `Q`
    /// Process Noise spectral density mapping to Velocity variance.
    pub Q_velocity: f32,
    /// Process Noise spectral density mapping to Sensor Drift variance.
    pub Q_bias: f32,
}

impl Default for KalmanFilterXY {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(missing_docs)]
impl KalmanFilterXY {
    const M11: usize = Matrix2x2f32::M11;
    const M22: usize = Matrix2x2f32::M22;

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

impl KalmanFilterXY {
    /// Constructor.
    #[allow(non_snake_case)]
    #[must_use]
    pub fn new() -> Self {
        let mut P = Matrix3x3xM2x2f32::default();
        // Seed initial Position uncertainty (ie, confident within 1 meter)
        P[Self::PP][Self::M11] = 1.0;
        P[Self::PP][Self::M22] = 1.0;

        // Seed initial Velocity uncertainty (ie, confident within 0.5 m/s)
        P[Self::VV][Self::M11] = 0.25;
        P[Self::VV][Self::M22] = 0.25;

        // Seed initial Bias uncertainty (ie, accelerometer bias)
        P[Self::BB][Self::M11] = 0.01;
        P[Self::BB][Self::M22] = 0.01;

        Self {
            pos: Vector2f32 { x: 0.0, y: 0.0 },
            vel: Vector2f32 { x: 0.0, y: 0.0 },
            acc_bias: Vector2f32 { x: 0.0, y: 0.0 },
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

impl KalmanFilterXY {
    #[inline]
    #[must_use]
    pub fn pos(&self) -> Vector2f32 {
        self.pos
    }

    #[inline]
    #[must_use]
    pub fn vel(&self) -> Vector2f32 {
        self.vel
    }

    #[inline]
    #[must_use]
    pub fn acc_bias(&self) -> Vector2f32 {
        self.acc_bias
    }
}

// **** Predict ****

impl KalmanFilterXY {
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
    pub fn predict_state(&mut self, acc_measurement: Vector2f32, dt: f32) {
        // Physical mechanics
        // s = ut + 0.5 * a * t²
        self.pos += (self.vel + 0.5 * acc_measurement * dt) * dt;
        // v = u + a * t
        self.vel += acc_measurement * dt;
        // Bias remains constant during prediction, it is modeled as a random walk in covariance.
        // So `self.acc_bias` not updated.
    }

    #[allow(non_snake_case)]
    pub fn predict_covariance(&mut self, dt: f32) {
        // Capture the current a posteriori state (P_k₋₁).
        let P_old = self.P;

        // =====================================================================
        // PROPAGATE THE COVARIANCE (F * P * F^T)
        // =====================================================================

        let dt2 = dt * dt;
        // --- POSITION COLUMNS ---
        self.P[Self::PP] = P_old[Self::PP] + (P_old[Self::VP] + P_old[Self::PV]) * dt + P_old[Self::VV] * dt2;
        self.P[Self::VP] = P_old[Self::VP] + (P_old[Self::VV] - P_old[Self::BP]) * dt - P_old[Self::BV] * dt2;
        self.P[Self::BP] = P_old[Self::BP] + P_old[Self::BV] * dt;

        // --- VELOCITY COLUMNS ---
        self.P[Self::PV] = P_old[Self::PV] + (P_old[Self::VV] - P_old[Self::PB]) * dt - P_old[Self::VB] * dt2;
        self.P[Self::VV] = P_old[Self::VV] - (P_old[Self::BV] + P_old[Self::VB]) * dt + P_old[Self::BB] * dt2;
        self.P[Self::BV] = P_old[Self::BV] - P_old[Self::BB] * dt;

        // --- BIAS COLUMNS ---
        self.P[Self::PB] = P_old[Self::PB] + P_old[Self::VB] * dt;
        self.P[Self::VB] = P_old[Self::VB] - P_old[Self::BB] * dt;
        self.P[Self::BB] = P_old[Self::BB];

        // =====================================================================
        // APPLY PROCESS NOISE (Q)
        // =====================================================================
        // Continuous process noise integrated over dt maps to the diagonal variance slots of Velocity and Bias.

        // Standard continuous noise integration layout
        self.P[Self::VV].add_diagonal_scalar_in_place(self.Q_velocity * dt);
        self.P[Self::BB].add_diagonal_scalar_in_place(self.Q_bias * dt);

        // Time propagation is highly sensitive to asymmetric shearing, so enforce symmetry on the covariance matrix.
        self.P.enforce_symmetry();
    }
}

// **** Correct ***

impl KalmanFilterXY {
    #[allow(non_snake_case)]
    pub fn correct_position_delayed(&mut self, position: Vector2f32, past_pos: Vector2f32, R: Vector2f32) {
        // Calculate the 3x3 Innovation Covariance matrix: S = H * P * H^T + R
        let S = self.P[Self::PP].add_diagonal_vector(R);

        // The the innovation matrix may be non-invertible.
        // This happens very rarely and is due to rounding errors when the process noise covariance `Q` is small.
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
        let error = position - past_pos;

        // Update the physical state vectors
        self.pos += K_pos * error;
        self.vel += K_vel * error;
        self.acc_bias += K_acc_bias * error;

        // Extract the submatrices representing H * P.
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

    #[allow(non_snake_case)]
    pub fn correct_position(&mut self, position: Vector2f32, R: Vector2f32) {
        self.correct_position_delayed(position, self.pos, R);
    }

    /// Joseph's Stabilized Form for the covariance update step:
    /// P{k} = (I - KH)* P_{k-1} *(I - KH)^T + KRK^T).
    /// While computationally more expensive, it guarantees the result remains positive-definite.
    /// That is, it ensures the covariance matrix has positive eigenvalues and remains valid and invertible for future updates.
    #[allow(non_snake_case)]
    pub fn correct_position_joseph(&mut self, position: Vector2f32, R: Vector2f32) {
        // Calculate the 3x3 Innovation Covariance matrix: S = H * P * H^T + R
        let S = self.P[Self::PP].add_diagonal_vector(R);

        // The the innovation matrix may be non-invertible.
        // This happens very rarely and is due to rounding errors when the process noise covariance `Q` is small.
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
        let I_minus_K_pos_t = Matrix2x2f32::identity() - K_pos_t; // Simplified distribution

        // Calculate the columns of intermediate matrix A = (I - KH)P
        let A = Matrix3x3xM2x2::from_column_array([
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

#[cfg(test)]
mod test_traits {
    use super::*;

    fn _is_normal<T: Sized + Send + Sync + Unpin>() {}
    fn is_full<T: Sized + Send + Sync + Unpin + Copy + Clone + Default + PartialEq>() {}

    #[test]
    fn normal_types() {
        is_full::<KalmanFilterXY>();
    }
}

#[cfg(test)]
mod tests_position {
    use super::*; // Pulls in PositionKalmanFilter, Vector3f32, and Matrix3x3xM3x3f32

    #[test]
    fn test_2d_position_convergence_and_maneuver() {
        // Initialize the filter with your tuned parameter states
        let mut filter = KalmanFilterXY::new();

        // Capture initial uncertainty variance values from the main diagonal blocks
        let init_p_pos_x = filter.P[KalmanFilterXY::PP][KalmanFilterXY::M11];
        //let init_p_vel_x = filter.P[PositionKalmanFilter::VV][PositionKalmanFilter::M11];

        // Setup simulation pacing parameters
        let dt = 0.01; // 100 Hz tracking thread loop

        // Define realistic GPS/UWB measurement noise variances (R)
        // Tracks ±1.5m horizontal variance and ±3.0m vertical variance
        let r_gps = Vector2f32 { x: 2.25, y: 2.25 };

        // Define the ground-truth physical state parameters
        let mut true_pos = Vector2f32 { x: 10.0, y: -5.0 }; // Start at an offset
        let mut true_vel = Vector2f32::default();
        let true_bias = Vector2f32 { x: -0.03, y: 0.04 }; // System accelerometer bias

        // Prime the filter's initial position guess to match our starting point
        filter.pos = true_pos;

        println!("\n--- PHASE 1: STATIONARY GPS LOCK (1 SECOND) ---");
        #[allow(clippy::cast_possible_truncation)]
        let loops_stationary = (1.0 / dt) as i32;
        for _ in 0..loops_stationary {
            // Raw IMU reading = Kinematic Acc (0) + Bias
            let acc_measurement = Vector2f32::default() + true_bias;

            filter.predict_state(acc_measurement, dt);
            filter.predict_covariance(dt);

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
        let post_stat_p_pos_x = filter.P[KalmanFilterXY::PP][KalmanFilterXY::M11];
        assert!(post_stat_p_pos_x < init_p_pos_x, "Position covariance failed to contract under GPS track!");

        println!("\n--- PHASE 2: 2D DYNAMIC SLIDE MANEUVER (2 SECONDS) ---");
        // Simulate a diagonal lateral acceleration profile (+X, +Y)
        #[allow(clippy::cast_possible_truncation)]
        let loops_maneuver = (2.0 / dt) as i32;
        let true_acc_kinematic = Vector2f32 { x: 1.5, y: 2.0 };

        for _ in 0..loops_maneuver {
            // Update truth kinematics using the exact model equations (Trapezoidal Rule)
            true_pos += (true_vel + 0.5 * true_acc_kinematic * dt) * dt;
            true_vel += true_acc_kinematic * dt;

            // Raw IMU specific force generation
            let acc_measurement = true_acc_kinematic + true_bias;

            filter.predict_state(acc_measurement, dt);
            filter.predict_covariance(dt);

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

        println!("\n✅ 2D position state propagation and measurement updates verified successfully!");
    }
}
