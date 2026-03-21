use core::ops::{Add, Div, Mul, Neg, Sub};
use num_traits::{One, Zero};

use crate::sensor_fusion::{SensorFusion, q_dot};
use imu_sensors::ImuReading;
use vector_quaternion_matrix::{MathFunctions, Quaternion, Vector3d};

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
    T: Copy
        + One
        + Zero
        + Neg<Output = T>
        + PartialEq
        + PartialOrd
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + MathFunctions,
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
    T: Copy
        + One
        + Zero
        + Neg<Output = T>
        + PartialEq
        + PartialOrd
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + MathFunctions,
{
    fn set_free_parameters(&mut self, parameter0: T, _parameter1: T) {
        self.alpha = parameter0;
    }
    fn requires_initialization() -> bool {
        false
    }
    fn update_orientation(&mut self, imu_reading: ImuReading<T>, delta_t: T) -> Quaternion<T> {
        // Calculate quaternion derivative (qDot) from angular rate https://ahrs.readthedocs.io/en/latest/filters/angular.html#quaternion-derivative
        // Twice the actual value is used to reduce the number of multiplications needed
        let q_dot = q_dot(&self.q, imu_reading.gyro_rps);

        // Update the attitude quaternion using simple Euler integration (qNew = qOld + qDot*deltaT).
        // Note: to reduce the number of multiplications, _2qDot and halfDeltaT are used, ie qNew = qOld +_2qDot*deltaT*0.5.
        self.q += q_dot * delta_t;

        // use the normalized accelerometer data to calculate an estimate of the attitude
        let acc: Vector3d<T> = imu_reading.acc.normalized();
        let roll_radians: T = ComplementaryFilter::roll_radians_from_acc_normalized(acc);
        let pitch_radians = ComplementaryFilter::pitch_radians_from_acc_normalized(acc);
        let q: Quaternion<T> = Quaternion::from_roll_pitch_angles_radians(roll_radians, pitch_radians);

        // use a complementary filter to combine the gyro attitude estimate(q) with the accelerometer attitude estimate(a)
        self.q = (self.q - q) * self.alpha + q; // optimized form of `self.alpha * q + (1.0 - self.alpha) * q` : uses fewer operations and can take advantage of multiply-add instruction

        // normalize the orientation quaternion
        self.q.normalize();

        self.q
    }
}

#[cfg(any(debug_assertions, test))]
mod tests {
    #![allow(unused)]
    use super::*;
    use imu_sensors::ImuReadingf32;
    use vector_quaternion_matrix::{Quaternionf32, Vector3df32};

    fn is_normal<T: Sized + Send + Sync + Unpin>() {}

    #[test]
    fn normal_types() {
        is_normal::<ComplementaryFilter<f32>>();
    }
    #[test]
    fn update_orientation() {
        let mut sensor_fusion = ComplementaryFilterf32::default();
        let requires_initialization = ComplementaryFilterf32::requires_initialization();
        sensor_fusion.set_alpha(1.0);
        assert_eq!(requires_initialization, false);
        let delta_t: f32 = 0.0;
        let imu_reading = ImuReadingf32::default();
        let orientation: Quaternionf32 = sensor_fusion.update_orientation(imu_reading, delta_t);
        assert_eq!(orientation, Quaternion { w: 1.0, x: 0.0, y: 0.0, z: 0.0 })
    }
}
