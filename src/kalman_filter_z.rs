use num_traits::{ConstOne, ConstZero, float::FloatCore};
use vqm::{MathMethods, Matrix3x3, Matrix3x3Math, Vector3};

/// `f32` variant of `AltitudeKalmanFilter`.
pub type KalmanFilterZf32 = KalmanFilterZ<f32>;
/// `f64` variant of `AltitudeKalmanFilter`.
pub type KalmanFilterZf64 = KalmanFilterZ<f64>;

pub trait KalmanFilterZConstants {
    const ONE_HUNDRED: Self;
    const ONE_TENTH: Self;
    const ONE_HUNDREDTH: Self;
}

impl KalmanFilterZConstants for f32 {
    const ONE_HUNDRED: Self = 100.0;
    const ONE_TENTH: Self = 0.1;
    const ONE_HUNDREDTH: Self = 0.01;
}

impl KalmanFilterZConstants for f64 {
    const ONE_HUNDRED: Self = 100.0;
    const ONE_TENTH: Self = 0.1;
    const ONE_HUNDREDTH: Self = 0.01;
}

#[allow(non_snake_case)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KalmanFilterZ<T> {
    predicted: [T; 3],
    estimated: [T; 3],
    bias: T,
    /// Predicted System Uncertainty Covariance Matrix (P).
    P: Matrix3x3<T>,

    Q_velocity: T,
    Q_bias: T,
}

impl<T> Default for KalmanFilterZ<T>
where
    T: Copy + ConstZero + ConstOne + FloatCore + Matrix3x3Math + KalmanFilterZConstants,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> KalmanFilterZ<T>
where
    T: Copy + ConstOne + KalmanFilterZConstants,
{
    /// Q, process noise covariance matrix.
    const Q1: T = T::ONE_HUNDREDTH;
    const Q3: T = T::ONE;
}
impl<T> KalmanFilterZ<T> {
    /// indices to access matrix rows.
    const VELOCITY: usize = 0;
    const ALTITUDE: usize = 1;
    const BIAS: usize = 2;
}

impl<T> KalmanFilterZ<T>
where
    T: Copy + ConstZero + ConstOne + FloatCore + Matrix3x3Math + KalmanFilterZConstants,
{
    /// Constructor.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            predicted: [T::ZERO; 3],
            estimated: [T::ZERO; 3],
            bias: T::ZERO,
            Q_velocity: Self::Q1,
            Q_bias: Self::Q3,
            P: Matrix3x3::ZERO,
        }
    }
}

#[allow(non_snake_case)]
impl<T> KalmanFilterZ<T>
where
    T: Copy + ConstZero + ConstOne + FloatCore + MathMethods + Matrix3x3Math + KalmanFilterZConstants,
{
    /// Initializer targeting steady-state baseline parameters.
    pub fn new_steady_state(initial_altitude: T, Q_velocity: T, Q_bias: T, r_barometer: T) -> Self {
        // Calculate analytical steady-state variance bounds.
        // Higher sensor noise (R) increases state uncertainty boundaries.
        // Higher process noise (Q) indicates dynamic, fast-changing states.
        let steady_state_alt_variance = (Q_velocity * r_barometer).sqrt();
        let steady_state_vel_variance = Q_velocity;
        let steady_state_bias_variance = Q_bias;

        // Map variances to the diagonal elements of the Covariance Matrices
        let initial_covariance = Matrix3x3::from_diagonal_array([
            steady_state_vel_variance,
            steady_state_alt_variance,
            steady_state_bias_variance,
        ]);

        Self {
            estimated: [T::ZERO, initial_altitude, T::ZERO],
            predicted: [T::ZERO, initial_altitude, T::ZERO],
            P: initial_covariance,
            bias: T::ONE_TENTH, // Damping factor configuration baseline
            Q_velocity,
            Q_bias,
        }
    }

    pub fn set_velocity(&mut self, velocity: T) {
        self.estimated[0] = velocity;
    }

    pub fn reset(&mut self) {
        self.P = Matrix3x3::ONE * T::ONE_HUNDRED;
    }

    /// Returns doublet `(estimated velocity, estimated altitude)`.
    pub fn state(&self) -> (T, T) {
        (self.estimated[0], self.estimated[1])
    }
}

// **** Predict ****

#[allow(non_snake_case)]
impl<T> KalmanFilterZ<T>
where
    T: Copy + ConstZero + ConstOne + FloatCore + MathMethods + Matrix3x3Math,
{
    /// Phase 1: Predict state forward using IMU/Physics
    /// Call this at the IMU frequency or fixed control loop rate.
    pub fn predict(&mut self, acc: T, dt: T) -> [T; 3] {
        // States are a 3d vector with components: velocity, altitude, and bias.
        // Destructure the state vectors as references with meaningful names, for code legibility (Zero cost abstraction).
        //let Vector3 { x: estimated_velocity, y: estimated_altitude, z: estimated_bias } = self.estimated;
        //let Vector3 { x: ref mut predicted_velocity, y: ref mut predicted_altitude, z: ref mut predicted_bias } =
        //  self.predicted;

        // Kinematic Euler integration for velocity and altitude.
        self.predicted[Self::VELOCITY] = self.estimated[Self::VELOCITY] + (acc - self.estimated[Self::BIAS]) * dt;
        self.predicted[Self::ALTITUDE] = self.estimated[Self::ALTITUDE] + self.estimated[Self::VELOCITY] * dt;
        self.predicted[Self::BIAS] = self.estimated[Self::BIAS] + self.estimated[Self::BIAS] * (self.bias * dt);

        // State Transition Matrix (A)
        #[rustfmt::skip]
        let A = Matrix3x3::new([
            T::ONE,  T::ZERO, -dt,
            dt,      T::ONE,  T::ZERO,
            T::ZERO, T::ZERO, T::ONE + self.bias * dt,
        ]);

        // Process Noise Matrix (Q)
        let dt2 = dt * dt;
        let Q = Matrix3x3::from_diagonal_array([dt2 * self.Q_velocity, T::ZERO, dt2 * self.Q_bias]);

        // Project error covariance: P_new = A * P * A^T + Q
        self.P = A * self.P * A.transpose() + Q;

        // Safety: If no measurement arrives, the estimate tracks the prediction
        self.estimated = self.predicted;

        self.predicted
    }
}

// **** Correct ***

impl<T> KalmanFilterZ<T>
where
    T: Copy + ConstZero + ConstOne + FloatCore + Matrix3x3Math + KalmanFilterZConstants,
{
    /// Phase 2 Altitude Correction using new measurement.
    #[allow(non_snake_case)]
    pub fn correct_altitude(&mut self, altitude: T, R: T) {
        const M22: usize = 4;
        // H vector for altitude: [0, 1, 0]
        let H_transpose = Vector3 { x: T::ZERO, y: T::ONE, z: T::ZERO };

        // Innovation covariance: S = H * P * H^T + R
        let S = self.P[M22] + R;
        if S.abs() < T::epsilon() {
            return;
        }

        // Kalman Gain: K = P * H^T / S
        let K = (self.P * H_transpose) * (T::ONE / S);

        // Update state estimate
        let error = altitude - self.predicted[Self::ALTITUDE];
        let K_error = <[T; 3]>::from(K * error);

        self.estimated[Self::VELOCITY] = self.predicted[Self::VELOCITY] + K_error[Self::VELOCITY];
        self.estimated[Self::ALTITUDE] = self.predicted[Self::ALTITUDE] + K_error[Self::ALTITUDE];
        self.estimated[Self::BIAS] = self.predicted[Self::BIAS] + K_error[Self::BIAS];

        // Update error covariance: P = (I - KH)P, ie P -= (KH)P
        self.P -= K.outer_product(self.P.row(Self::ALTITUDE));

        // Prepare for next cycle if multiple corrections happen sequentially
        self.predicted = self.estimated;
    }
}

#[cfg(test)]
mod test_traits {
    use super::*;

    fn is_full<T: Sized + Send + Sync + Unpin + Copy + Clone + Default + PartialEq>() {}

    #[test]
    fn normal_types() {
        is_full::<KalmanFilterZf32>();
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use vqm::{Matrix3x3f32, Vector3f32};

    #[test]
    fn test_new() {
        let _kalman_filter = KalmanFilterZf32::new();
    }

    #[allow(non_snake_case)]
    #[test]
    fn kalman_covariance_update() {
        // Initialize the Kalman Gain vector (K)
        let K = Vector3f32 { x: 3.0, y: 7.0, z: 13.0 };

        // Initialize a starting Covariance Matrix (P)
        // We set the 2nd row to [2.0, 5.0, 11.0] to match our proven outer product values
        let P = Matrix3x3f32::new([
            10.0, 20.0, 30.0, // Row 1
            2.0, 5.0, 11.0, // Row 2 (altitude row)
            50.0, 60.0, 70.0, // Row 3
        ]);

        // Extract altitude row from the P matrix
        let altitude_row = P.row(KalmanFilterZf32::ALTITUDE);
        assert_eq!(Vector3f32 { x: 2.0, y: 5.0, z: 11.0 }, altitude_row);

        // Calculate the updated Covariance Matrix P_new.
        let K_HP = K.outer_product(altitude_row);

        let P_new = P - K_HP;

        // Calculate the mathematically expected output data layout:
        // Row 1: [10, 20, 30] - [6,  15, 33]  = [4,   5,  -3]
        // Row 2: [2,  5,  11] - [14, 35, 77]  = [-12, -30, -66]
        // Row 3: [50, 60, 70] - [26, 65, 143] = [24,  -5,  -73]
        assert_eq!(P_new, Matrix3x3f32::new([4.0, 5.0, -3.0, -12.0, -30.0, -66.0, 24.0, -5.0, -73.0]));
    }
}
