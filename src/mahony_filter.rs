use num_traits::{ConstOne, ConstZero, float::FloatCore};

use crate::{SensorFusion, SensorFusionMath};
use vqm::{MathConstants, Quaternion, QuaternionMath, SqrtMethods, TrigonometricMethods, Vector3};

/// Mahony filter for `f32`<br>
pub type MahonyFilterf32 = MahonyFilter<f32>;
/// Mahony filter for `f64`<br><br>
pub type MahonyFilterf64 = MahonyFilter<f64>;

/// [Mahony filter](https://ahrs.readthedocs.io/en/latest/filters/mahony.html).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MahonyFilter<T> {
    /// orientation quaternion.
    q: Quaternion<T>,
    kp: T,
    ki: T,
    error_integral: Vector3<T>,
    gyro_rps_1: Vector3<T>,
    gyro_rps_2: Vector3<T>,
    use_quadratic_interpolation: bool,
    use_matrix_exponential_approximation: bool,
}

impl<T> Default for MahonyFilter<T>
where
    T: Copy + ConstZero + ConstOne + MathConstants,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> MahonyFilter<T>
where
    T: Copy + ConstZero + ConstOne + MathConstants,
{
    /// Constructor.
    #[must_use]
    pub const fn new() -> Self {
        MahonyFilter {
            q: Quaternion { w: T::ONE, x: T::ZERO, y: T::ZERO, z: T::ZERO },
            kp: T::TEN,
            ki: T::ZERO,
            error_integral: Vector3 { x: T::ZERO, y: T::ZERO, z: T::ZERO },
            gyro_rps_1: Vector3 { x: T::ZERO, y: T::ZERO, z: T::ZERO },
            gyro_rps_2: Vector3 { x: T::ZERO, y: T::ZERO, z: T::ZERO },
            use_quadratic_interpolation: false,
            use_matrix_exponential_approximation: false,
        }
    }
}

impl<T> MahonyFilter<T>
where
    T: Copy + FloatCore + TrigonometricMethods + MathConstants + SqrtMethods + QuaternionMath + SensorFusionMath,
{
    pub fn set_proportional_integral(&mut self, kp: T, ki: T) {
        self.set_gains(kp, ki);
    }
}

impl<T> SensorFusion<T> for MahonyFilter<T>
where
    T: Copy + FloatCore + TrigonometricMethods + MathConstants + SqrtMethods + QuaternionMath + SensorFusionMath,
{
    fn set_gains(&mut self, gain0: T, gain1: T) {
        self.kp = gain0;
        self.ki = gain1;
    }
    fn gains(&self) -> (T, T) {
        (self.kp, self.ki)
    }
    fn requires_initialization() -> bool {
        true
    }

    /// Fuses accelerometer and gyroscope readings to give the orientation quaternion.
    fn fuse_acc_gyro(&mut self, acc: Vector3<T>, gyro_rps: Vector3<T>, delta_t: T) -> Quaternion<T> {
        // Normalize acceleration
        let acc = acc.normalize();

        // Calculate estimated direction of gravity in the sensor coordinate frame
        let gravity = self.q.gravity();

        // Error is the cross product between direction measured by acceleration and estimated direction of gravity
        let error = acc.cross(gravity);

        // Quadratic Interpolation (From Attitude Representation and Kinematic Propagation for Low-Cost UAVs by Robert T. Casey, Equation 14)
        // See https://docs.rosflight.org/v1.3/algorithms/estimator/#modifications-to-original-passive-filter for a publicly available explanation
        let mut gyro = gyro_rps;
        if self.use_quadratic_interpolation {
            gyro = gyro_rps * (T::FIVE / T::TWELVE) + self.gyro_rps_1 * (T::EIGHT / T::TWELVE)
                - self.gyro_rps_2 * (T::one() / T::TWELVE);
            self.gyro_rps_2 = self.gyro_rps_1;
            self.gyro_rps_1 = gyro_rps;
        }

        // Apply proportional feedback
        gyro += error * self.kp;

        // Apply integral feedback if ki set
        //if self.ki > T::zero() {
        self.error_integral += error * (self.ki * delta_t); // note brackets to ensure scalar multiplication is performed before vector multiplication
        gyro += self.error_integral;
        //}

        let q_dot = SensorFusionMath::derivative(self.q, gyro);
        if self.use_matrix_exponential_approximation {
            // Matrix Exponential Approximation (From Attitude Representation and Kinematic Propagation for Low-Cost UAVs by Robert T. Casey, Equation 12)
            let gyro_magnitude = gyro.norm();
            let theta = gyro_magnitude * T::HALF * delta_t;
            let (sin, cos) = theta.sin_cos();
            let t1 = cos;
            let t2 = (T::TWO / gyro_magnitude) * sin;

            self.q = self.q * t1 + q_dot * t2;
        } else {
            // Update the attitude quaternion using simple Euler integration
            self.q += q_dot * delta_t;
        }

        // normalize the orientation quaternion and return it
        self.q = self.q.normalize();
        self.q
    }

    fn fuse_acc_gyro_mag(
        &mut self,
        acc: Vector3<T>,
        gyro_rps: Vector3<T>,
        _mag: Vector3<T>,
        delta_t: T,
    ) -> Quaternion<T> {
        self.fuse_acc_gyro(acc, gyro_rps, delta_t)
    }

    fn correct_yaw_with_gain(&mut self, _yaw_radians: T, _gain: T, _delta_t: T) -> Quaternion<T> {
        self.q
    }
}

#[cfg(test)]
mod test_traits {
    use super::*;

    fn is_full<T: Sized + Send + Sync + Unpin + Copy + Clone + Default + PartialEq>() {}

    #[test]
    fn normal_types() {
        is_full::<MahonyFilter<f32>>();
    }
}

#[cfg(any(debug_assertions, test))]
mod tests {
    #![allow(unused)]
    use super::*;
    use vqm::Vector3f32;

    #[test]
    fn update_orientation() {
        let mut sensor_fusion = MahonyFilterf32::default();
        let requires_initialization = MahonyFilterf32::requires_initialization();
        assert!(requires_initialization);

        sensor_fusion.set_proportional_integral(10.0, 0.0);

        let delta_t: f32 = 0.0;
        let acc = Vector3f32::default();
        let gyro_rps = Vector3f32::default();

        let orientation = sensor_fusion.fuse_acc_gyro(acc, gyro_rps, delta_t);
        assert_eq!(orientation, Quaternion { w: 1.0, x: 0.0, y: 0.0, z: 0.0 });
    }
}
