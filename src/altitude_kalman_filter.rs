use num_traits::{ConstOne, ConstZero, float::FloatCore};
use vqm::{MathMethods, Matrix3x3, Matrix3x3Math, Vector3};

/// `f32` variant of `AltitudeKalmanFilter`.
pub type AltitudeKalmanFilterf32 = AltitudeKalmanFilter<f32>;
/// `f64` variant of `AltitudeKalmanFilter`.
pub type AltitudeKalmanFilterf64 = AltitudeKalmanFilter<f64>;

pub trait AltitudeKalmanFilterConstants {
    const ONE_HUNDRED: Self;
    const ONE_TENTH: Self;
    const ONE_HUNDREDTH: Self;
}

impl AltitudeKalmanFilterConstants for f32 {
    const ONE_HUNDRED: Self = 100.0;
    const ONE_TENTH: Self = 0.1;
    const ONE_HUNDREDTH: Self = 0.01;
}

impl AltitudeKalmanFilterConstants for f64 {
    const ONE_HUNDRED: Self = 100.0;
    const ONE_TENTH: Self = 0.1;
    const ONE_HUNDREDTH: Self = 0.01;
}

#[allow(non_snake_case)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AltitudeKalmanFilter<T> {
    predicted: Vector3<T>,
    estimated: Vector3<T>,
    beta: T,
    /// Predicted System Uncertainty Covariance Matrix (P).
    P: Matrix3x3<T>,
    /// Estimated Post-Correction Error Covariance Matrix (E).
    E: Matrix3x3<T>,

    // --- Hyperparameters & Tuning Constants ---
    q_velocity: T,
    q_bias: T,
}

impl<T> Default for AltitudeKalmanFilter<T>
where
    T: Copy + ConstZero + ConstOne + FloatCore + Matrix3x3Math + AltitudeKalmanFilterConstants,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> AltitudeKalmanFilter<T>
where
    T: Copy + ConstOne + AltitudeKalmanFilterConstants,
{
    /// Q, process noise covariance matrix.
    const Q1: T = T::ONE_HUNDREDTH;
    const Q3: T = T::ONE;

    /// R, measurement covariance matrix.
    const _R: f32 = 0.004 * 0.004;
    /// indices to access matrix rows.
    const _VELOCITY_ROW: usize = 0;
    const ALTITUDE_ROW: usize = 1;
    const _BIAS_ROW: usize = 2;
}

impl<T> AltitudeKalmanFilter<T>
where
    T: Copy + ConstZero + ConstOne + FloatCore + Matrix3x3Math + AltitudeKalmanFilterConstants,
{
    /// Constructor.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            predicted: Vector3::ZERO,
            estimated: Vector3::ZERO,
            beta: T::ZERO,
            q_velocity: Self::Q1,
            q_bias: Self::Q3,
            E: Matrix3x3::ZERO,
            P: Matrix3x3::ZERO,
        }
    }
}

impl<T> AltitudeKalmanFilter<T>
where
    T: Copy + ConstZero + ConstOne + FloatCore + MathMethods + Matrix3x3Math + AltitudeKalmanFilterConstants,
{
    /// Initializer targeting steady-state baseline parameters.
    pub fn new_steady_state(initial_altitude: T, q_velocity: T, q_bias: T, r_barometer: T) -> Self {
        // Calculate analytical steady-state variance bounds.
        // Higher sensor noise (R) increases state uncertainty boundaries.
        // Higher process noise (Q) indicates dynamic, fast-changing states.
        let steady_state_alt_variance = (q_velocity * r_barometer).sqrt();
        let steady_state_vel_variance = q_velocity;
        let steady_state_bias_variance = q_bias;

        // Map variances to the diagonal elements of the Covariance Matrices
        #[rustfmt::skip]
        let initial_covariance = Matrix3x3::new([
            steady_state_vel_variance, T::ZERO,                   T::ZERO,
            T::ZERO,                   steady_state_alt_variance, T::ZERO,
            T::ZERO,                   T::ZERO,                   steady_state_bias_variance,
        ]);

        Self {
            estimated: Vector3 { x: T::ZERO, y: initial_altitude, z: T::ZERO },
            predicted: Vector3 { x: T::ZERO, y: initial_altitude, z: T::ZERO },
            P: initial_covariance,
            E: initial_covariance,
            beta: T::ONE_TENTH, // Damping factor configuration baseline
            q_velocity,
            q_bias,
        }
    }

    pub fn set_velocity(&mut self, velocity: T) {
        self.estimated.x = velocity;
    }

    pub fn reset(&mut self) {
        self.E = Matrix3x3::ONE * T::ONE_HUNDRED;
    }

    /// Returns doublet `(estimated velocity, estimated altitude)`.
    pub fn state(&self) -> (T, T) {
        (self.estimated.x, self.estimated.y)
    }
}

// **** Predict ****

impl<T> AltitudeKalmanFilter<T>
where
    T: Copy + ConstZero + ConstOne + FloatCore + MathMethods + Matrix3x3Math,
{
    /// Phase 1: Predict state forward using IMU/Physics
    /// Call this at your IMU frequency or fixed control loop rate.
    #[allow(non_snake_case)]
    #[rustfmt::skip]
    pub fn predict(&mut self, acceleration_measurement: T, delta_t: T) -> Vector3<T> {
        // States are a 3d vector with components: velocity, altitude, and bias.
        // Destructure the state vectors as references with meaningful names, for code legibility (Zero cost abstraction).
        let Vector3 { x: estimated_velocity, y: estimated_altitude, z: estimated_bias } = self.estimated;
        let Vector3 { x: ref mut predicted_velocity, y: ref mut predicted_altitude, z: ref mut predicted_bias } =
            self.predicted;

        // Kinematic Euler integration for velocity and altitude.
        *predicted_velocity = estimated_velocity + (acceleration_measurement - estimated_bias) * delta_t;
        *predicted_altitude = estimated_altitude + estimated_velocity * delta_t;
        *predicted_bias = estimated_bias * (T::ONE + self.beta * delta_t);

        // State Transition Matrix (A)
        let A = Matrix3x3::new([
            T::ONE,  T::ZERO, -delta_t,
            delta_t, T::ONE,  T::ZERO,
            T::ZERO, T::ZERO, T::ONE + self.beta * delta_t,
        ]);

        // Process Noise Matrix (Q)
        let dt2 = delta_t * delta_t;
        let Q = Matrix3x3::new([
            dt2 * self.q_velocity, T::ZERO, T::ZERO, // Fixed negative sign from original code if standard variance
            T::ZERO,               T::ZERO, T::ZERO,
            T::ZERO,               T::ZERO, dt2 * self.q_bias,
        ]);

        // Project error covariance: P = A * E * A^T + Q
        self.P = (A * self.E * A.transpose()) + Q;

        // Safety: If no measurement arrives, the estimate tracks tje prediction
        self.estimated = self.predicted;
        self.E = self.P;

        self.predicted
    }
}

// **** Correct ***

impl<T> AltitudeKalmanFilter<T>
where
    T: Copy + ConstZero + ConstOne + FloatCore + Matrix3x3Math + AltitudeKalmanFilterConstants,
{
    /// Phase 2 Altitude Correction using new measurement.
    #[allow(non_snake_case)]
    pub fn correct_altitude(&mut self, altitude: T, R: T) {
        const M22: usize = 4;
        // H vector for altitude: [0, 1, 0]
        let H_transpose = Vector3 { x: T::ZERO, y: T::ONE, z: T::ZERO };

        // Innovation covariance: S = H * P * H^T + R
        let S = self.P[M22] + R;

        // Kalman Gain: K = P * H^T / S
        let K = (self.P * H_transpose) * (T::ONE / S);

        // Update state estimate
        let predicted_altitude = self.predicted.y;

        let error = altitude - predicted_altitude;
        self.estimated = self.predicted + K * error;

        // Update error covariance: E = (I - KH)P
        self.E = self.P - K.outer_product(self.P.row(Self::ALTITUDE_ROW));

        // Prepare for next cycle if multiple corrections happen sequentially
        self.predicted = self.estimated;
        self.P = self.E;
    }
}

#[cfg(test)]
mod test_traits {
    use super::*;

    fn is_full<T: Sized + Send + Sync + Unpin + Copy + Clone + Default + PartialEq>() {}

    #[test]
    fn normal_types() {
        is_full::<AltitudeKalmanFilterf32>();
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use vqm::{Matrix3x3f32, Vector3f32};

    #[test]
    fn test_new() {
        let _kalman_filter = AltitudeKalmanFilterf32::new();
    }

    #[test]
    fn kalman_covariance_update() {
        // Initialize the Kalman Gain vector (K)
        let k = Vector3f32 { x: 3.0, y: 7.0, z: 13.0 };

        // Initialize a starting Covariance Matrix (P)
        // We set the 2nd row to [2.0, 5.0, 11.0] to match our proven outer product values
        let p = Matrix3x3f32::new([
            10.0, 20.0, 30.0, // Row 1
            2.0, 5.0, 11.0, // Row 2 (altitude row)
            50.0, 60.0, 70.0, // Row 3
        ]);

        // Extract altitude row from the P matrix
        let altitude_row = p.row(AltitudeKalmanFilterf32::ALTITUDE_ROW);
        assert_eq!(Vector3f32 { x: 2.0, y: 5.0, z: 11.0 }, altitude_row);

        // Calculate the updated Covariance Matrix (E).
        let kh_p = k.outer_product(altitude_row);

        let e = p - kh_p;

        // Calculate the mathematically expected output data layout:
        // Row 1: [10, 20, 30] - [6,  15, 33]  = [4,   5,  -3]
        // Row 2: [2,  5,  11] - [14, 35, 77]  = [-12, -30, -66]
        // Row 3: [50, 60, 70] - [26, 65, 143] = [24,  -5,  -73]
        assert_eq!(e, Matrix3x3f32::new([4.0, 5.0, -3.0, -12.0, -30.0, -66.0, 24.0, -5.0, -73.0]));
    }
}
