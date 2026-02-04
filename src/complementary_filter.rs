#![allow(unused)]
use crate::sensor_fusion::{SensorFusion, q_dot};
use vector_quaternion_matrix::{Quaternion, Vector3d};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ComplementaryFilter {
    // orientation quaternion
    q: Quaternion,
    acc_magnitude_squared_max: f32,
    alpha: f32,
}

impl Default for ComplementaryFilter {
    fn default() -> Self {
        ComplementaryFilter {
            q: Quaternion {
                w: 1.0,
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            acc_magnitude_squared_max: 4.0,
            alpha: 0.95,
        }
    }
}

impl ComplementaryFilter {
    /// Calculate roll (theta) from the normalized accelerometer readings
    pub fn roll_radians_from_acc_normalized(acc: Vector3d) -> f32 {
        (acc.y).atan2(acc.z)
    }
    /// Calculate pitch (phi) from the normalized accelerometer readings
    pub fn pitch_radians_from_acc_normalized(acc: Vector3d) -> f32 {
        (-acc.x).atan2((acc.y * acc.y + acc.z * acc.z).sqrt())
    }
    pub fn set_alpha(&mut self, alpha: f32) {
        self.set_free_parameters(alpha, 0.0);
    }
}

impl SensorFusion for ComplementaryFilter {
    fn set_free_parameters(&mut self, parameter0: f32, _parameter1: f32) {
        self.alpha = parameter0;
    }
    fn requires_initialization() -> bool {
        false
    }
    fn update_orientation(
        &mut self,
        gyro_rps: &Vector3d,
        accelerometer: &Vector3d,
        delta_t: f32,
    ) -> Quaternion {
        // Calculate quaternion derivative (qDot) from angular rate https://ahrs.readthedocs.io/en/latest/filters/angular.html#quaternion-derivative
        // Twice the actual value is used to reduce the number of multiplications needed
        let q_dot = q_dot(&self.q, &gyro_rps);

        // Update the attitude quaternion using simple Euler integration (qNew = qOld + qDot*deltaT).
        // Note: to reduce the number of multiplications, _2qDot and halfDeltaT are used, ie qNew = qOld +_2qDot*deltaT*0.5.
        self.q += q_dot * delta_t;

        // use the normalized accelerometer data to calculate an estimate of the attitude
        let acc: Vector3d = accelerometer.normalized();
        let a = Quaternion::from_roll_pitch_angles_radians(
            ComplementaryFilter::roll_radians_from_acc_normalized(acc),
            ComplementaryFilter::pitch_radians_from_acc_normalized(acc),
        );

        // use a complementary filter to combine the gyro attitude estimate(q) with the accelerometer attitude estimate(a)
        self.q = (self.q - a) * self.alpha + a; // optimized form of `self.alpha * q + (1.0 - self.alpha) * a` : uses fewer operations and can take advantage of multiply-add instruction

        // normalize the orientation quaternion
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
        is_normal::<ComplementaryFilter>();
    }
    #[test]
    fn update_orientation() {
        let mut sensor_fusion = ComplementaryFilter::default();
        let requires_initialization = ComplementaryFilter::requires_initialization();
        sensor_fusion.set_alpha(0.9);
        assert_eq!(requires_initialization, false);
        let gyro_rps = Vector3d::default();
        let acc = Vector3d::default();
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
