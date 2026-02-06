#![allow(unused)]

use crate::sensor_fusion::{SensorFusion, q_dot};
use vector_quaternion_matrix::{MathMethods, Quaternion, Vector3d};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MadgwickFilter {
    // orientation quaternion
    q: Quaternion,
    acc_magnitude_squared_max: f32,
    beta: f32,
}

impl Default for MadgwickFilter {
    fn default() -> Self {
        MadgwickFilter {
            q: Quaternion {
                w: 1.0,
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            acc_magnitude_squared_max: 4.0,
            beta: 1.0,
        }
    }
}

impl MadgwickFilter {
    pub fn set_beta(&mut self, beta: f32) {
        self.set_free_parameters(beta, 0.0);
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
impl SensorFusion for MadgwickFilter {
    fn set_free_parameters(&mut self, parameter0: f32, _parameter1: f32) {
        self.beta = parameter0;
    }

    fn requires_initialization() -> bool {
        true
    }

    fn update_orientation(
        &mut self,
        gyro_rps: &Vector3d,
        acc: &Vector3d,
        delta_t: f32,
    ) -> Quaternion {
        // Calculate quaternion derivative (q_dot, aka dq/dt) from the angular rate
        let mut q_dot = q_dot(&self.q, gyro_rps);

        // Acceleration is an unreliable indicator of orientation when in high-g maneuvers,
        // so exclude it from the calculation in these cases
        let acc_magnitude_squared = acc.squared_norm();
        if acc_magnitude_squared <= self.acc_magnitude_squared_max {
            // Normalize acceleration if it is non-zero
            let mut a = *acc;
            if acc_magnitude_squared != 0.0 {
                a *= acc_magnitude_squared.reciprocal_sqrt();
            }
            // make copies of the components of q to simplify the algebraic expressions
            let q0 = self.q.w;
            let q1 = self.q.x;
            let q2 = self.q.y;
            let q3 = self.q.z;
            // Auxiliary variables to avoid repeated arithmetic
            let _2q1q1_plus_2q2q2 = 2.0 * (q1 * q1 + q2 * q2);
            let common = 2.0 * (q0 * q0 + q3 * q3 - 1.0 + _2q1q1_plus_2q2q2 + a.z);

            // Gradient decent algorithm corrective step
            let mut step = Quaternion {
                w: q0 * (_2q1q1_plus_2q2q2) + q2 * a.x - q1 * a.y,
                x: q1 * common - q3 * a.x - q0 * a.y,
                y: q2 * common + q0 * a.x - q3 * a.y,
                z: q3 * (_2q1q1_plus_2q2q2) - q1 * a.x - q2 * a.y,
            };
            step.normalize();
            // Subtract the corrective step from the quaternion derivative
            q_dot -= self.beta * step;
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
    use super::*;

    fn is_normal<T: Sized + Send + Sync + Unpin>() {}

    #[test]
    fn normal_types() {
        is_normal::<MadgwickFilter>();
    }
    #[test]
    fn update_orientation() {
        let mut sensor_fusion: MadgwickFilter = MadgwickFilter::default();
        let requires_initialization = MadgwickFilter::requires_initialization();
        assert_eq!(requires_initialization, true);
        sensor_fusion.set_beta(1.0);
        let gyro_rps: Vector3d = Vector3d::default();
        let acc: Vector3d = Vector3d::default();
        let delta_t: f32 = 0.0;
        let orientation: Quaternion = sensor_fusion.update_orientation(&gyro_rps, &acc, delta_t);
        assert_eq!(
            orientation,
            Quaternion {
                w: 1.0,
                x: 0.0,
                y: 0.0,
                z: 0.0
            }
        )
    }
}
