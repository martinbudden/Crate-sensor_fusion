#![allow(unused)]

use crate::sensor_fusion::{SensorFusion, q_dot};
use vector_quaternion_matrix::{Quaternion, Vector3d};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MahonyFilter {
    // orientation quaternion
    q: Quaternion,
    kp: f32,
    ki: f32,
    error_integral: Vector3d,
    gyro_rps_1: Vector3d,
    gyro_rps_2: Vector3d,
    use_quadratic_interpolation: bool,
    use_matrix_exponential_approximation: bool,
}

impl Default for MahonyFilter {
    fn default() -> Self {
        MahonyFilter {
            q: Quaternion {
                w: 1.0,
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            kp: 10.0,
            ki: 0.0,
            error_integral: Vector3d::default(),
            gyro_rps_1: Vector3d::default(),
            gyro_rps_2: Vector3d::default(),
            use_quadratic_interpolation: false,
            use_matrix_exponential_approximation: false,
        }
    }
}

impl MahonyFilter {
    pub fn set_proportional_integral(&mut self, kp: f32, ki: f32) {
        self.set_free_parameters(kp, ki);
    }
}

impl SensorFusion for MahonyFilter {
    fn set_free_parameters(&mut self, parameter0: f32, parameter1: f32) {
        self.kp = parameter0;
        self.ki = parameter1;
    }
    fn requires_initialization() -> bool {
        true
    }
    fn update_orientation(
        &mut self,
        gyro_rps: &Vector3d,
        accelerometer: &Vector3d,
        delta_t: f32,
    ) -> Quaternion {
        // Normalize acceleration
        let acc = accelerometer.normalized();

        // Calculate estimated direction of gravity in the sensor coordinate frame
        let gravity = self.q.gravity();

        // Error is the cross product between direction measured by acceleration and estimated direction of gravity
        let error = acc.cross(gravity);

        // Quadratic Interpolation (From Attitude Representation and Kinematic Propagation for Low-Cost UAVs by Robert T. Casey, Equation 14)
        // See https://docs.rosflight.org/v1.3/algorithms/estimator/#modifications-to-original-passive-filter for a publicly available explanation
        let mut gyro = *gyro_rps;
        if self.use_quadratic_interpolation {
            gyro = *gyro_rps * (5.0 / 12.0) + self.gyro_rps_1 * (8.0 / 12.0)
                - self.gyro_rps_2 * (1.0 / 12.0);
            self.gyro_rps_2 = self.gyro_rps_1;
            self.gyro_rps_1 = *gyro_rps;
        }

        // Apply proportional feedback
        gyro += error * self.kp;

        // Apply integral feedback if ki set
        if self.ki > 0.0 {
            self.error_integral += error * (self.ki * delta_t); // note brackets to ensure scalar multiplication is performed before vector multiplication
            gyro += self.error_integral;
        }

        if self.use_matrix_exponential_approximation {
            // Matrix Exponential Approximation (From Attitude Representation and Kinematic Propagation for Low-Cost UAVs by Robert T. Casey, Equation 12)
            let gyro_magnitude = gyro.norm();
            let theta = gyro_magnitude * 0.5 * delta_t;
            let (sin, cos) = theta.sin_cos();
            let t1 = cos;
            let t2 = (1.0 / gyro_magnitude) * sin;

            self.q.w =
                t1 * self.q.w + t2 * (-gyro.x * self.q.x - gyro.y * self.q.y - gyro.z * self.q.z);
            self.q.x =
                t1 * self.q.x + t2 * (gyro.x * self.q.w + gyro.z * self.q.y - gyro.y * self.q.z);
            self.q.y =
                t1 * self.q.y + t2 * (gyro.y * self.q.w - gyro.z * self.q.x + gyro.x * self.q.z);
            self.q.z =
                t1 * self.q.z + t2 * (gyro.z * self.q.w + gyro.y * self.q.x - gyro.x * self.q.y);
        } else {
            let q_dot = q_dot(&self.q, &gyro);
            // Update the attitude quaternion using simple Euler integration
            self.q += q_dot * delta_t;
        }

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
        is_normal::<MahonyFilter>();
    }
    #[test]
    fn update_orientation() {
        let mut sensor_fusion = MahonyFilter::default();
        let requires_initialization = MahonyFilter::requires_initialization();
        sensor_fusion.set_proportional_integral(10.0, 0.0);
        assert_eq!(requires_initialization, true);
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
