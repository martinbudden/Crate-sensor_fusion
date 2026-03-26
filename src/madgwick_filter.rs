use crate::sensor_fusion::{SensorFusion, q_dot};
use core::ops::{Div, Neg, Sub};
use num_traits::{One, Zero};
use vector_quaternion_matrix::{MathMethods, Quaternion, Vector3d};

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

    fn fuse_acc_gyro(&mut self, acc: Vector3d<T>, gyro_rps: Vector3d<T>, delta_t: T) -> Quaternion<T> {
        // Calculate quaternion derivative (q_dot, aka dq/dt) from the angular rate
        let mut q_dot = q_dot(&self.q, gyro_rps);

        // Acceleration is an unreliable indicator of orientation when in high-g maneuvers,
        // so exclude it from the calculation in these cases
        let acc_magnitude_squared = acc.squared_norm();
        if acc_magnitude_squared <= self.acc_magnitude_squared_max {
            // Normalize acceleration if it is non-zero
            let mut a = acc;
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

    fn fuse_acc_gyro_mag(
        &mut self,
        acc: Vector3d<T>,
        gyro_rps: Vector3d<T>,
        mag: Vector3d<T>,
        delta_t: T,
    ) -> Quaternion<T> {
        let mut a = acc;
        let acc_magnitude_squared = a.squared_norm();
        // Acceleration is an unreliable indicator of orientation when in high-g maneuvers,
        // so exclude it from the calculation in these cases
        if acc_magnitude_squared > self.acc_magnitude_squared_max {
            a.set_zero();
        }

        let mut m = mag;
        m.normalize();

        // make copies of the components of q to simplify the algebraic expressions
        let q0 = self.q.w;
        let q1 = self.q.x;
        let q2 = self.q.y;
        let q3 = self.q.z;
        // Auxiliary variables to avoid repeated arithmetic
        let q0q0 = q0 * q0;
        let q0q1 = q0 * q1;
        let q0q2 = q0 * q2;
        let q0q3 = q0 * q3;
        let q1q1 = q1 * q1;
        let q1q2 = q1 * q2;
        let q1q3 = q1 * q3;
        let q2q2 = q2 * q2;
        let q2q3 = q2 * q3;
        let q3q3 = q3 * q3;

        let q1q1_plus_q2q2 = q1q1 + q2q2;
        let q2q2_plus_q3q3 = q2q2 + q3q3;

        // Reference direction of Earth's magnetic field
        let two = T::one() + T::one();
        let hx = m.x * (q0q0 + q1q1 - q2q2_plus_q3q3) + two * (m.y * (q1q2 - q0q3) + m.z * (q0q2 + q1q3));
        let hy = two * (m.x * (q0q3 + q1q2) + m.y * (q0q0 - q1q1 + q2q2 - q3q3) + m.z * (q2q3 - q0q1));

        let bx_bx = hx * hx + hy * hy;
        let bx = bx_bx.sqrt();
        let bz = two * (m.x * (q1q3 - q0q2) + m.y * (q0q1 + q2q3)) + m.z * (q0q0 - q1q1_plus_q2q2 + q3q3);
        let bz_bz = bz * bz;
        let _4bx_bz = two * two * bx * bz;

        let mx_bx = m.x * bx;
        let my_bx = m.y * bx;
        let mz_bx = m.z * bx;
        let mz_bz = m.z * bz;

        let ax_plus_mx_bz = a.x + m.x * bz;
        let ay_plus_my_bz = a.y + m.y * bz;

        let sum_squares_minus_one = q0q0 + q1q1_plus_q2q2 + q3q3 - T::one();
        let common = sum_squares_minus_one + q1q1_plus_q2q2 + a.z;

        let half = T::one() / two;
        // Gradient decent algorithm corrective step
        let s0 = q0 * two * (q1q1_plus_q2q2 * (T::one() + bz_bz) + bx_bx * q2q2_plus_q3q3) - q1 * ay_plus_my_bz
            + q2 * (ax_plus_mx_bz - mz_bx)
            + q3 * (my_bx - _4bx_bz * q0q1);

        let s1 = -q0 * ay_plus_my_bz
            + q1 * two * (common + mz_bz + bx_bx * q2q2_plus_q3q3 + bz_bz * (sum_squares_minus_one + q1q1_plus_q2q2))
            - q2 * my_bx
            - q3 * (ax_plus_mx_bz + mz_bx + _4bx_bz * (half * sum_squares_minus_one + q1q1));

        let s2 = q0 * (ax_plus_mx_bz - mz_bx) - q1 * my_bx
            + q2 * two
                * (common
                    + mz_bz
                    + mx_bx
                    + bx_bx * (sum_squares_minus_one + q2q2_plus_q3q3)
                    + bz_bz * (sum_squares_minus_one + q1q1_plus_q2q2))
            - q3 * (ay_plus_my_bz + _4bx_bz * q1q2);

        let s3 = q0 * my_bx
            - q1 * (ax_plus_mx_bz + mz_bx + _4bx_bz * (half * sum_squares_minus_one + q3q3))
            - q2 * ay_plus_my_bz
            + q3 * two
                * (q1q1_plus_q2q2 * (T::one() + bz_bz) + mx_bx + bx_bx * (sum_squares_minus_one + q2q2_plus_q3q3));

        let mut step = Quaternion { w: s0, x: s1, y: s2, z: s3 };
        step.normalize();

        let mut q_dot = q_dot(&self.q, gyro_rps);
        q_dot -= step * self.beta;

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
        let mut madgwick_filter = MadgwickFilterf32::default();
        let requires_initialization = MadgwickFilterf32::requires_initialization();
        assert_eq!(requires_initialization, true);

        madgwick_filter.set_beta(1.0);

        let delta_t: f32 = 0.0;
        let acc = Vector3df32::default();
        let gyro_rps = Vector3df32::default();

        let orientation = madgwick_filter.fuse_acc_gyro(acc, gyro_rps, delta_t);
        assert_eq!(orientation, Quaternion { w: 1.0, x: 0.0, y: 0.0, z: 0.0 })
    }
}
