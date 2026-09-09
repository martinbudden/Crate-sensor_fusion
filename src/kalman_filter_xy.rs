use vqm::{Matrix2x2f32, Matrix3x3xM2x2, Matrix3x3xM2x2f32, Vector2f32};

/// `f32` variant of `PositionKalmanFilterXY`.
pub type KalmanFilterXYf32 = KalmanFilterXY;

/// Linear Kalman Filter in 2 dimensions.
#[allow(non_snake_case)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KalmanFilterXY {
    /// 2D Kinematic State Vectors.
    pub state: KalmanStateXY,

    // Predicted System Uncertainty Covariance Matrix (P).
    /// `P`: Prediction error covariance (aka state covariance matrix, the system's internal uncertainty).
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
            state: KalmanStateXY::new(),
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
        self.state.pos
    }

    #[inline]
    #[must_use]
    pub fn vel(&self) -> Vector2f32 {
        self.state.vel
    }

    #[inline]
    #[must_use]
    pub fn acc_bias(&self) -> Vector2f32 {
        self.state.acc_bias
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
    ///
    /// pos_k = pos_k₋₁ + vel_k₋₁ * dT + 0.5 * acc * dT²
    /// vel_k = vel_k₋₁ + acc * dT
    /// ```
    pub fn predict_state(&mut self, acc_measurement: Vector2f32, dt: f32) {
        // Physical mechanics
        // s = ut + 0.5 * a * t²
        self.state.pos += (self.state.vel + 0.5 * acc_measurement * dt) * dt;
        // v = u + a * t
        self.state.vel += acc_measurement * dt;
        // Bias remains constant during prediction, it is modeled as a random walk in covariance.
        // So `self.acc_bias` not updated.
    }

    /// Predict covariance:
    ///
    /// P = F * P * Fᵀ + Q.
    ///
    /// P: State covariance matrix
    /// F: State transition matrix
    /// Q: Process noise covariance.
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
impl KalmanFilterXY {
    /// Covariance correction step.
    ///
    /// `S = H * P * Hᵀ + R`
    /// `K = P * Hᵀ * S⁻¹`
    /// `P = (I - K * H) * P`.
    ///
    ///
    /// R is measurement noise covariance.
    pub fn correct_position_standard_form(
        &mut self,
        measurement: Vector2f32,
        predicted_measurement: Vector2f32,
        R: Vector2f32,
    ) {
        // Extract the submatrices representing H * P.
        let P_pos = self.P[Self::PP];
        let P_vel = self.P[Self::PV];
        let P_acc_bias = self.P[Self::PB];

        // Calculate S, the Residual Covariance matrix:
        // S = H * P * Hᵀ + R
        let S = P_pos.add_diagonal_vector(R);

        // The the Residual Covariance matrix may be non-invertible.
        // This happens very rarely and is due to rounding errors when the process noise covariance Q is small.
        let Some(S_inv) = S.try_inverse() else {
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
        measurement: Vector2f32,
        predicted_measurement: Vector2f32,
        R: Vector2f32,
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

        // Precompute transposes and helper terms
        let K_pos_t = K_pos.transpose();
        let K_vel_t = K_vel.transpose();
        let K_acc_bias_t = K_acc_bias.transpose();
        let I_minus_K_pos_t = Matrix2x2f32::identity() - K_pos_t; // Simplified distribution

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

    pub fn correct_position(&mut self, position: Vector2f32, R: Vector2f32) {
        self.correct_position_standard_form(position, self.state.pos, R);
    }

    pub fn correct_position_joseph(&mut self, position: Vector2f32, R: Vector2f32) {
        self.correct_position_joseph_stabilized_form(position, self.state.pos, R);
    }
}

#[allow(unused)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KalmanStateXY {
    // 3D Kinematic State Vectors
    /// Position {x, y}.
    pub pos: Vector2f32,
    /// Velocity {x, y}.
    pub vel: Vector2f32,
    /// Accelerometer Bias {x, y}.
    pub acc_bias: Vector2f32,
}

impl Default for KalmanStateXY {
    fn default() -> Self {
        Self::new()
    }
}

impl KalmanStateXY {
    /// Constructor.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            pos: Vector2f32 { x: 0.0, y: 0.0 },
            vel: Vector2f32 { x: 0.0, y: 0.0 },
            acc_bias: Vector2f32 { x: 0.0, y: 0.0 },
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
        is_full::<KalmanFilterXY>();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_2d_position_convergence() {
        let mut filter = KalmanFilterXY::new();

        // Capture initial uncertainty variance values from the main diagonal blocks
        let init_p_pos_x = filter.P[KalmanFilterXY::PP][KalmanFilterXY::M11];
        //let init_p_vel_x = filter.P[PositionKalmanFilter::VV][PositionKalmanFilter::M11];

        // Setup simulation pacing parameters
        let dt = 0.01; // 100 Hz tracking thread loop

        // Define realistic GPS/UWB measurement noise variances (R)
        // Tracks ±1.5m horizontal variance
        let r_gps = Vector2f32 { x: 2.25, y: 2.25 };

        // Define the ground-truth physical state parameters
        let mut true_pos = Vector2f32 { x: 10.0, y: -5.0 }; // Start at an offset
        let mut true_vel = Vector2f32::default();
        let true_bias = Vector2f32 { x: -0.03, y: 0.04 }; // System accelerometer bias

        // Prime the filter's initial position guess to match our starting point
        filter.state.pos = true_pos;

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
        println!("Stationary -> True Pos X: {:.4}, Estimated Pos X: {:.4}", true_pos.x, filter.pos().x);
        println!("Stationary -> True Vel X: {:.4}, Estimated Vel X: {:.4}", true_vel.x, filter.vel().x);
        println!("Stationary -> True Bias X: {:.4}, Estimated Bias X: {:.4}", true_bias.x, filter.acc_bias().x);

        // Verification assertions for Phase 1
        assert!((filter.pos().x - true_pos.x).abs() < 0.05, "Position drifted while stationary!");
        assert!((filter.vel().x - true_vel.x).abs() < 0.05, "Velocity accumulated noise while stationary!");

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
        println!("Maneuver -> True Final Pos X: {:.4}, Estimated Final Pos X: {:.4}", true_pos.x, filter.pos().x);
        println!("Maneuver -> True Final Pos Y: {:.4}, Estimated Final Pos Y: {:.4}", true_pos.y, filter.pos().y);
        println!("Maneuver -> True Final Vel X: {:.4}, Estimated Final Vel X: {:.4}", true_vel.x, filter.vel().x);
        println!("Maneuver -> True Final Vel Y: {:.4}, Estimated Final Vel Y: {:.4}", true_vel.y, filter.vel().y);

        // Tracking precision assertions
        assert!((filter.pos().x - true_pos.x).abs() < 0.1, "Filter missed true physical X position track!");
        assert!((filter.pos().y - true_pos.y).abs() < 0.1, "Filter missed true physical Y position track!");
        assert!((filter.vel().x - true_vel.x).abs() < 0.1, "Filter velocity tracking tracking failed on X axis!");
        assert!((filter.vel().y - true_vel.y).abs() < 0.1, "Filter velocity tracking tracking failed on Y axis!");

        println!("\n✅ 2D position state propagation and measurement updates verified successfully!");
    }
}
