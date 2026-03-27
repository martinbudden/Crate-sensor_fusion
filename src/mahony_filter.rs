use core::ops::{Div, Neg, Sub};
use num_traits::{One, Zero};

use crate::sensor_fusion::{SensorFusion, q_dot};
use vector_quaternion_matrix::{MathConstants, MathMethods, Quaternion, Vector3d};

pub type MahonyFilterf32 = MahonyFilter<f32>;
pub type MahonyFilterf64 = MahonyFilter<f64>;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MahonyFilter<T> {
    q: Quaternion<T>, // orientation quaternion
    kp: T,
    ki: T,
    error_integral: Vector3d<T>,
    gyro_rps_1: Vector3d<T>,
    gyro_rps_2: Vector3d<T>,
    use_quadratic_interpolation: bool,
    use_matrix_exponential_approximation: bool,
}

impl<T> Default for MahonyFilter<T>
where
    T: Zero + One + Default + MathConstants,
{
    fn default() -> Self {
        MahonyFilter {
            q: Quaternion::default(),
            kp: T::TEN,
            ki: T::zero(),
            error_integral: Vector3d::default(),
            gyro_rps_1: Vector3d::default(),
            gyro_rps_2: Vector3d::default(),
            use_quadratic_interpolation: false,
            use_matrix_exponential_approximation: false,
        }
    }
}

impl<T> MahonyFilter<T>
where
    T: Copy
        + One
        + Zero
        + Neg<Output = T>
        + PartialEq
        + PartialOrd
        + Sub<Output = T>
        + Div<Output = T>
        + MathMethods
        + MathConstants,
{
    pub fn set_proportional_integral(&mut self, kp: T, ki: T) {
        self.set_free_parameters(kp, ki);
    }
}

impl<T> SensorFusion<T> for MahonyFilter<T>
where
    T: Copy
        + One
        + Zero
        + Neg<Output = T>
        + PartialOrd
        + Sub<Output = T>
        + Div<Output = T>
        + MathMethods
        + MathConstants,
{
    fn set_free_parameters(&mut self, parameter0: T, parameter1: T) {
        self.kp = parameter0;
        self.ki = parameter1;
    }
    fn requires_initialization() -> bool {
        true
    }

    /// Fuses accelerometer and gyroscope readings to give the orientation quaternion.
    fn fuse_acc_gyro(&mut self, acc: Vector3d<T>, gyro_rps: Vector3d<T>, delta_t: T) -> Quaternion<T> {
        // Normalize acceleration
        let acc = acc.normalized();

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
        if self.ki > T::zero() {
            self.error_integral += error * (self.ki * delta_t); // note brackets to ensure scalar multiplication is performed before vector multiplication
            gyro += self.error_integral;
        }

        if self.use_matrix_exponential_approximation {
            // Matrix Exponential Approximation (From Attitude Representation and Kinematic Propagation for Low-Cost UAVs by Robert T. Casey, Equation 12)
            let gyro_magnitude = gyro.norm();
            let theta = gyro_magnitude * T::HALF * delta_t;
            let (sin, cos) = theta.sin_cos();
            let t1 = cos;
            let t2 = (T::one() / gyro_magnitude) * sin;

            self.q.w = t1 * self.q.w + t2 * (-gyro.x * self.q.x - gyro.y * self.q.y - gyro.z * self.q.z);
            self.q.x = t1 * self.q.x + t2 * (gyro.x * self.q.w + gyro.z * self.q.y - gyro.y * self.q.z);
            self.q.y = t1 * self.q.y + t2 * (gyro.y * self.q.w - gyro.z * self.q.x + gyro.x * self.q.z);
            self.q.z = t1 * self.q.z + t2 * (gyro.z * self.q.w + gyro.y * self.q.x - gyro.x * self.q.y);
        } else {
            let q_dot = q_dot(&self.q, gyro);
            // Update the attitude quaternion using simple Euler integration
            self.q += q_dot * delta_t;
        }

        // Normalize the orientation quaternion
        self.q.normalize();

        self.q
    }

    fn fuse_acc_gyro_mag(
        &mut self,
        acc: Vector3d<T>,
        gyro_rps: Vector3d<T>,
        _mag: Vector3d<T>,
        delta_t: T,
    ) -> Quaternion<T> {
        self.fuse_acc_gyro(acc, gyro_rps, delta_t)
    }
}

#[cfg(any(debug_assertions, test))]
mod tests {
    #![allow(unused)]
    use super::*;
    use vector_quaternion_matrix::Vector3df32;

    fn is_normal<T: Sized + Send + Sync + Unpin>() {}
    fn is_full<T: Sized + Send + Sync + Unpin + Copy + Clone + Default + PartialEq>() {}

    #[test]
    fn normal_types() {
        is_full::<MahonyFilter<f32>>();
    }
    #[test]
    fn update_orientation() {
        let mut sensor_fusion = MahonyFilterf32::default();
        let requires_initialization = MahonyFilterf32::requires_initialization();
        assert_eq!(requires_initialization, true);

        sensor_fusion.set_proportional_integral(10.0, 0.0);

        let delta_t: f32 = 0.0;
        let acc = Vector3df32::default();
        let gyro_rps = Vector3df32::default();

        let orientation = sensor_fusion.fuse_acc_gyro(acc, gyro_rps, delta_t);
        assert_eq!(orientation, Quaternion { w: 1.0, x: 0.0, y: 0.0, z: 0.0 })
    }
}
