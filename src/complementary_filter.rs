use core::ops::Neg;
use num_traits::{One, Zero, float::FloatCore};

use crate::{SensorFusion, SensorFusionMath};
use vqm::{Quaternion, QuaternionMath, SqrtMethods, TrigonometricMethods, Vector3};

/// Complementary filter for `f32`<br>
pub type ComplementaryFilterf32 = ComplementaryFilter<f32>;
/// Complementary filter for `f64`<br><br>
pub type ComplementaryFilterf64 = ComplementaryFilter<f64>;

/// [Complementary filter](https://ahrs.readthedocs.io/en/latest/filters/complementary.html).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ComplementaryFilter<T> {
    /// orientation quaternion.
    q: Quaternion<T>,
    acc_magnitude_squared_max: T,
    alpha: T,
}

impl<T> Default for ComplementaryFilter<T>
where
    T: Copy + One + Zero,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> ComplementaryFilter<T>
where
    T: Copy + One + Zero,
{
    /// Constructor.
    #[must_use]
    pub fn new() -> Self {
        ComplementaryFilter {
            q: Quaternion::default(),
            acc_magnitude_squared_max: T::one() + T::one() + T::one() + T::one(),
            alpha: T::one(),
        }
    }
}

impl<T> ComplementaryFilter<T>
where
    T: Copy + One + Zero + Neg<Output = T> + TrigonometricMethods + SqrtMethods,
{
    /// Calculate roll (theta) from the normalized accelerometer readings.
    pub fn roll_radians_from_acc_normalized(acc: Vector3<T>) -> T {
        (acc.y).atan2(acc.z)
    }
    /// Calculate pitch (phi) from the normalized accelerometer readings.
    pub fn pitch_radians_from_acc_normalized(acc: Vector3<T>) -> T {
        (-acc.x).atan2((acc.y * acc.y + acc.z * acc.z).sqrt())
    }
    pub fn set_alpha(&mut self, alpha: T) {
        self.alpha = alpha;
    }
}

impl<T> SensorFusion<T> for ComplementaryFilter<T>
where
    T: Copy + FloatCore + TrigonometricMethods + SqrtMethods + QuaternionMath + SensorFusionMath,
{
    fn set_gains(&mut self, gain0: T, _gain1: T) {
        self.alpha = gain0;
    }
    fn gains(&self) -> (T, T) {
        (self.alpha, T::zero())
    }

    fn requires_initialization() -> bool {
        false
    }

    /// Fuses accelerometer and gyroscope readings to give the orientation quaternion.
    fn fuse_acc_gyro(&mut self, acc: Vector3<T>, gyro_rps: Vector3<T>, delta_t: T) -> Quaternion<T> {
        // Calculate quaternion derivative (qDot) from angular rate https://ahrs.readthedocs.io/en/latest/filters/angular.html#quaternion-derivative
        // Twice the actual value is used to reduce the number of multiplications needed
        let q_dot = SensorFusionMath::derivative(self.q, gyro_rps);

        // Update the attitude quaternion using simple Euler integration (qNew = qOld + qDot*deltaT).
        // Note: to reduce the number of multiplications, _2qDot and halfDeltaT are used, ie qNew = qOld +_2qDot*deltaT*0.5.
        self.q += q_dot * delta_t;

        // use the normalized accelerometer data to calculate an estimate of the attitude
        let acc: Vector3<T> = acc.normalize();
        let roll_radians: T = ComplementaryFilter::roll_radians_from_acc_normalized(acc);
        let pitch_radians = ComplementaryFilter::pitch_radians_from_acc_normalized(acc);
        let q: Quaternion<T> = Quaternion::from_roll_pitch_radians(roll_radians, pitch_radians);

        // use a complementary filter to combine the gyro attitude estimate(q) with the accelerometer attitude estimate(a)
        self.q = (self.q - q) * self.alpha + q; // optimized form of `self.alpha * q + (1.0 - self.alpha) * q` : uses fewer operations and can take advantage of multiply-add instruction

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
        is_full::<ComplementaryFilter<f32>>();
    }
}
#[cfg(any(debug_assertions, test))]
mod tests {
    #![allow(unused)]
    use super::*;
    use vqm::{Quaternionf32, Vector3f32};

    #[test]
    fn update_orientation() {
        let mut sensor_fusion = ComplementaryFilterf32::default();
        let requires_initialization = ComplementaryFilterf32::requires_initialization();
        assert!(!requires_initialization);

        sensor_fusion.set_alpha(1.0);

        let delta_t: f32 = 0.0;
        let acc = Vector3f32::default();
        let gyro_rps = Vector3f32::default();

        let orientation = sensor_fusion.fuse_acc_gyro(acc, gyro_rps, delta_t);

        assert_eq!(orientation, Quaternion { w: 1.0, x: 0.0, y: 0.0, z: 0.0 });
    }
}
