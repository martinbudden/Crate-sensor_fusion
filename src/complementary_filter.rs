use core::ops::{Div, Neg, Sub};
use num_traits::{One, Zero};

use crate::sensor_fusion::{SensorFusion, q_dot};
use vector_quaternion_matrix::{MathMethods, Quaternion, Vector3d};

pub type ComplementaryFilterf32 = ComplementaryFilter<f32>;
pub type ComplementaryFilterf64 = ComplementaryFilter<f64>;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ComplementaryFilter<T> {
    // orientation quaternion
    q: Quaternion<T>,
    acc_magnitude_squared_max: T,
    alpha: T,
}

impl<T> Default for ComplementaryFilter<T>
where
    T: One + Zero,
{
    fn default() -> Self {
        ComplementaryFilter {
            q: Quaternion::default(),
            acc_magnitude_squared_max: T::one() + T::one() + T::one() + T::one(),
            alpha: T::one(),
        }
    }
}

impl<T> ComplementaryFilter<T>
where
    T: Copy + One + Zero + Neg<Output = T> + PartialEq + PartialOrd + Sub<Output = T> + Div<Output = T> + MathMethods,
{
    /// Calculate roll (theta) from the normalized accelerometer readings
    pub fn roll_radians_from_acc_normalized(acc: Vector3d<T>) -> T {
        (acc.y).atan2(acc.z)
    }
    /// Calculate pitch (phi) from the normalized accelerometer readings
    pub fn pitch_radians_from_acc_normalized(acc: Vector3d<T>) -> T {
        (-acc.x).atan2((acc.y * acc.y + acc.z * acc.z).sqrt())
    }
    pub fn set_alpha(&mut self, alpha: T) {
        self.set_free_parameters(alpha, T::zero());
    }
}

impl<T> SensorFusion<T> for ComplementaryFilter<T>
where
    T: Copy + One + Zero + Neg<Output = T> + PartialOrd + Sub<Output = T> + Div<Output = T> + MathMethods,
{
    fn set_free_parameters(&mut self, parameter0: T, _parameter1: T) {
        self.alpha = parameter0;
    }
    fn requires_initialization() -> bool {
        false
    }

    fn fuse_acc_gyro(&mut self, acc: Vector3d<T>, gyro_rps: Vector3d<T>, delta_t: T) -> Quaternion<T> {
        // Calculate quaternion derivative (qDot) from angular rate https://ahrs.readthedocs.io/en/latest/filters/angular.html#quaternion-derivative
        // Twice the actual value is used to reduce the number of multiplications needed
        let q_dot = q_dot(&self.q, gyro_rps);

        // Update the attitude quaternion using simple Euler integration (qNew = qOld + qDot*deltaT).
        // Note: to reduce the number of multiplications, _2qDot and halfDeltaT are used, ie qNew = qOld +_2qDot*deltaT*0.5.
        self.q += q_dot * delta_t;

        // use the normalized accelerometer data to calculate an estimate of the attitude
        let acc: Vector3d<T> = acc.normalized();
        let roll_radians: T = ComplementaryFilter::roll_radians_from_acc_normalized(acc);
        let pitch_radians = ComplementaryFilter::pitch_radians_from_acc_normalized(acc);
        let q: Quaternion<T> = Quaternion::from_roll_pitch_angles_radians(roll_radians, pitch_radians);

        // use a complementary filter to combine the gyro attitude estimate(q) with the accelerometer attitude estimate(a)
        self.q = (self.q - q) * self.alpha + q; // optimized form of `self.alpha * q + (1.0 - self.alpha) * q` : uses fewer operations and can take advantage of multiply-add instruction

        // normalize the orientation quaternion
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
    use imu_sensors::ImuReadingf32;
    use vector_quaternion_matrix::{Quaternionf32, Vector3df32};

    fn is_normal<T: Sized + Send + Sync + Unpin>() {}
    fn is_full<T: Sized + Send + Sync + Unpin + Copy + Clone + Default + PartialEq>() {}

    #[test]
    fn normal_types() {
        is_full::<ComplementaryFilter<f32>>();
    }
    #[test]
    fn update_orientation() {
        let mut sensor_fusion = ComplementaryFilterf32::default();
        let requires_initialization = ComplementaryFilterf32::requires_initialization();
        assert_eq!(requires_initialization, false);

        sensor_fusion.set_alpha(1.0);

        let delta_t: f32 = 0.0;
        let acc = Vector3df32::default();
        let gyro_rps = Vector3df32::default();

        let orientation = sensor_fusion.fuse_acc_gyro(acc, gyro_rps, delta_t);

        assert_eq!(orientation, Quaternion { w: 1.0, x: 0.0, y: 0.0, z: 0.0 })
    }
}
