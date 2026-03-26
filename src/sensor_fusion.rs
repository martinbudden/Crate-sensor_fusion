use core::ops::{Add, Div, Mul, Sub};
use num_traits::One;
use vector_quaternion_matrix::{Quaternion, Quaternionf32, Vector3d};

pub trait SensorFusion<T> {
    fn set_free_parameters(&mut self, parameter0: T, parameter1: T);
    fn requires_initialization() -> bool;

    fn fuse_acc_gyro(&mut self, acc: Vector3d<T>, gyro_rps: Vector3d<T>, delta_t: T) -> Quaternion<T>;
    fn fuse_acc_gyro_mag(&mut self, acc: Vector3d<T>, gyro: Vector3d<T>, mag: Vector3d<T>, delta_t: T) -> Quaternion<T>;
}

/// Calculate quaternion derivative (dq/dt aka q_dot) from angular rate https://ahrs.readthedocs.io/en/latest/filters/angular.html#quaternion-derivative
pub fn q_dot<T>(q: &Quaternion<T>, gyro_rps: Vector3d<T>) -> Quaternion<T>
where
    T: Copy + One + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T>,
{
    let half = T::one() / (T::one() + T::one());
    Quaternion {
        w: (q.x * gyro_rps.x - q.y * gyro_rps.y - q.z * gyro_rps.z) * half,
        x: (q.w * gyro_rps.x + q.y * gyro_rps.z - q.z * gyro_rps.y) * half,
        y: (q.w * gyro_rps.y - q.x * gyro_rps.z + q.z * gyro_rps.x) * half,
        z: (q.w * gyro_rps.z + q.x * gyro_rps.y - q.y * gyro_rps.x) * half,
    }
}

#[cfg(any(debug_assertions, test))]
mod tests {
    use super::*;
    use vector_quaternion_matrix::Vector3df32;

    pub struct TestStruct;
    impl SensorFusion<f32> for TestStruct {
        fn set_free_parameters(&mut self, _parameter0: f32, _parameter1: f32) {}
        fn requires_initialization() -> bool {
            true
        }
        fn fuse_acc_gyro(&mut self, _acc: Vector3df32, _gyro_rps: Vector3df32, _delta_t:f32) -> Quaternionf32 {
            Quaternionf32::default()
        }
        fn fuse_acc_gyro_mag(&mut self, acc: Vector3df32, gyro_rps: Vector3df32, _mag: Vector3df32, delta_t: f32) -> Quaternionf32 {
            self.fuse_acc_gyro(acc, gyro_rps, delta_t)
        }
    }
    

    #[allow(dead_code)]
    fn sensor_fusion() {
        let mut test_struct: TestStruct = TestStruct {};
        TestStruct::requires_initialization();
        //assert_eq!(TestStruct::requires_initialization(), true);

        test_struct.set_free_parameters(0.0, 0.0);

        let delta_t: f32 = 0.0;
        let acc = Vector3df32::default();
        let gyro_rps = Vector3df32::default();

        let orientation = test_struct.fuse_acc_gyro(acc, gyro_rps, delta_t);
        assert_eq!(orientation, Quaternion::default());
    }
}
