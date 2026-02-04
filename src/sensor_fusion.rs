use vector_quaternion_matrix::{Quaternion, Vector3d};

pub trait SensorFusion {
    fn update_orientation(
        &mut self,
        gyro_rps: &Vector3d,
        acc: &Vector3d,
        delta_t: f32,
    ) -> Quaternion;
    fn set_free_parameters(&mut self, parameter0: f32, parameter1: f32);
    fn requires_initialization() -> bool;
}

/// Calculate quaternion derivative (dq/dt aka q_dot) from angular rate https://ahrs.readthedocs.io/en/latest/filters/angular.html#quaternion-derivative
pub fn q_dot(q: &Quaternion, gyro_rps: &Vector3d) -> Quaternion {
    Quaternion {
        w: (q.x * gyro_rps.x - q.y * gyro_rps.y - q.z * gyro_rps.z) * 0.5,
        x: (q.w * gyro_rps.x + q.y * gyro_rps.z - q.z * gyro_rps.y) * 0.5,
        y: (q.w * gyro_rps.y - q.x * gyro_rps.z + q.z * gyro_rps.x) * 0.5,
        z: (q.w * gyro_rps.z + q.x * gyro_rps.y - q.y * gyro_rps.x) * 0.5,
    }
}

#[cfg(any(debug_assertions, test))]
mod tests {
    use super::*;

    pub struct TestStruct;
    impl SensorFusion for TestStruct {
        fn set_free_parameters(&mut self, _parameter0: f32, _parameter1: f32) {}
        fn requires_initialization() -> bool {
            true
        }
        fn update_orientation(
            &mut self,
            _gyro_rps: &Vector3d,
            _accelerometer: &Vector3d,
            _delta_t: f32,
        ) -> Quaternion {
            Quaternion::default()
        }
    }

    #[allow(dead_code)]
    fn sensor_fusion() {
        let mut test_struct: TestStruct = TestStruct {};
        TestStruct::requires_initialization();
        //assert_eq!(TestStruct::requires_initialization(), true);

        test_struct.set_free_parameters(0.0, 0.0);

        let gyro_rps = Vector3d::default();
        let acc = Vector3d::default();
        let orientation = test_struct.update_orientation(&gyro_rps, &acc, 0.0);
        assert_eq!(orientation, Quaternion::default());
    }
}
