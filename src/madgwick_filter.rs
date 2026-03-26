use crate::sensor_fusion::{SensorFusion, q_dot};
use core::ops::{Div, Neg, Sub};
use imu_sensors::ImuReading;
use num_traits::{One, Zero};
use vector_quaternion_matrix::{MathMethods, Quaternion};

pub type MadgwickFilterf32 = MadgwickFilter<f32>;
pub type MadgwickFilterf64 = MadgwickFilter<f64>;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MadgwickFilter<T> {
    // orientation quaternion
    q: Quaternion<T>,
    acc_magnitude_squared_max: T,
    beta: T,
}

impl<T> Default for MadgwickFilter<T>
where
    T: Zero + One + Default,
{
    fn default() -> Self {
        MadgwickFilter {
            q: Quaternion::default(),
            acc_magnitude_squared_max: T::one() + T::one() + T::one() + T::one(),
            beta: T::one(),
        }
    }
}

impl<T> MadgwickFilter<T>
where
    T: Copy + One + Zero + Neg<Output = T> + PartialOrd + Sub<Output = T> + Div<Output = T> + MathMethods,
{
    pub fn set_beta(&mut self, beta: T) {
        self.set_free_parameters(beta, T::zero());
    }
}

/// Madgwick AHRS algorithm, calculates orientation by fusing output from gyroscope and accelerometer.
/// (No magnetometer is used in this implementation.)
///
/// The orientation is calculated as the integration of the gyroscope measurements summed with the measurement from the accelerometer multiplied by a gain.
/// A low gain gives more weight to the gyroscope more and so is more susceptible to drift.
/// A high gain gives more weight to the accelerometer and so is more susceptible to accelerometer noise, lag, and other accelerometer errors.
/// A gain of zero means that orientation is determined by solely by the gyroscope.
///
/// See [Sebastian Madgwick's Phd thesis](https://x-io.co.uk/downloads/madgwick-phd-thesis.pdf)
/// and also x-io Technologies [sensor fusion library](https://github.com/xioTechnologies/Fusion)
///
/// For computation efficiency this code refactors the code used in many implementations (Arduino, Adafruit, M5Stack, Reefwing-AHRS),
/// [see MadgwickRefactoring](../../../documents/MadgwickRefactoring.md)
///
impl<T> SensorFusion<T> for MadgwickFilter<T>
where
    T: Copy + One + Zero + Neg<Output = T> + PartialOrd + Sub<Output = T> + Div<Output = T> + MathMethods,
{
    fn set_free_parameters(&mut self, parameter0: T, _parameter1: T) {
        self.beta = parameter0;
    }

    fn requires_initialization() -> bool {
        true
    }

    fn update_orientation(&mut self, imu_reading: ImuReading<T>, delta_t: T) -> Quaternion<T> {
        // Calculate quaternion derivative (q_dot, aka dq/dt) from the angular rate
        let mut q_dot = q_dot(&self.q, imu_reading.gyro_rps);

        // Acceleration is an unreliable indicator of orientation when in high-g maneuvers,
        // so exclude it from the calculation in these cases
        let acc_magnitude_squared = imu_reading.acc.squared_norm();
        if acc_magnitude_squared <= self.acc_magnitude_squared_max {
            // Normalize acceleration if it is non-zero
            let mut a = imu_reading.acc;
            if acc_magnitude_squared != T::zero() {
                a *= acc_magnitude_squared.reciprocal_sqrt();
            }
            // make copies of the components of q to simplify the algebraic expressions
            let q0 = self.q.w;
            let q1 = self.q.x;
            let q2 = self.q.y;
            let q3 = self.q.z;
            // Auxiliary variables to avoid repeated arithmetic
            let two = T::one() + T::one();
            let _2q1q1_plus_2q2q2 = two * (q1 * q1 + q2 * q2);
            let common = two * (q0 * q0 + q3 * q3 - T::one() + _2q1q1_plus_2q2q2 + a.z);

            // Gradient decent algorithm corrective step
            let mut step = Quaternion {
                w: q0 * (_2q1q1_plus_2q2q2) + q2 * a.x - q1 * a.y,
                x: q1 * common - q3 * a.x - q0 * a.y,
                y: q2 * common + q0 * a.x - q3 * a.y,
                z: q3 * (_2q1q1_plus_2q2q2) - q1 * a.x - q2 * a.y,
            };
            step.normalize();
            // Subtract the corrective step from the quaternion derivative
            q_dot -= step * self.beta;
        }

        // Update the orientation quaternion using simple Euler integration
        self.q += q_dot * delta_t;
        // Normalize the orientation quaternion
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
    fn is_full<T: Sized + Send + Sync + Unpin + Copy + Clone + Default + PartialEq>() {}

    #[test]
    fn normal_types() {
        is_full::<MadgwickFilter<f32>>();
    }
    #[test]
    fn update_orientation() {
        let mut sensor_fusion = MadgwickFilterf32::default();
        let requires_initialization = MadgwickFilterf32::requires_initialization();
        assert_eq!(requires_initialization, true);
        sensor_fusion.set_beta(1.0);
        let delta_t: f32 = 0.0;
        let imu_reading = ImuReading::default();
        let orientation = sensor_fusion.update_orientation(imu_reading, delta_t);
        assert_eq!(orientation, Quaternion { w: 1.0, x: 0.0, y: 0.0, z: 0.0 })
    }
}
