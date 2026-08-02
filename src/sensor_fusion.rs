use vqm::{Quaternion, Quaternionf32, Vector3, Vector3f32};

/// Common interface for the sensor fusion filters (Madgwick, Mahony, complementary).
/// ```
/// use vqm::{Vector3f32,Quaternionf32};
/// use sensor_fusion::{MadgwickFilterf32,SensorFusion};
///
/// let mut madgwick_filter = MadgwickFilterf32::default();
///
/// let delta_t: f32 = 0.0;
/// let acc = Vector3f32::default();
/// let gyro_rps = Vector3f32::default();
///
/// let orientation = madgwick_filter.fuse_acc_gyro(acc, gyro_rps, delta_t);
/// assert_eq!(orientation, Quaternionf32 { w: 1.0, x: 0.0, y: 0.0, z: 0.0 });
/// ```
pub trait SensorFusion<T> {
    fn set_gains(&mut self, gain0: T, gain1: T);
    fn gains(&self) -> (T, T);

    fn requires_initialization() -> bool;

    fn fuse_acc_gyro(&mut self, acc: Vector3<T>, gyro_rps: Vector3<T>, delta_t: T) -> Quaternion<T>;
    fn fuse_acc_gyro_mag(&mut self, acc: Vector3<T>, gyro: Vector3<T>, mag: Vector3<T>, delta_t: T) -> Quaternion<T>;
    fn correct_yaw_with_gain(&mut self, yaw_radians: T, gain: T, delta_t: T) -> Quaternion<T>;
}

#[allow(unused)]
pub trait SensorFusionf32 {
    fn set_gains(&mut self, gain0: f32, gain1: f32);
    fn requires_initialization() -> bool;

    fn fuse_acc_gyro(&mut self, acc: Vector3f32, gyro_rps: Vector3f32, delta_t: f32) -> Quaternionf32;
    fn fuse_acc_gyro_mag(&mut self, acc: Vector3f32, gyro: Vector3f32, mag: Vector3f32, delta_t: f32) -> Quaternionf32;
    fn correct_yaw(&mut self, yaw_radians: f32, gain: f32, delta_t: f32) -> Quaternionf32;
}

#[allow(clippy::doc_paragraphs_missing_punctuation)]
/// Trait to allow sensor fusion filters to be used with method-call syntax, ie:<br>
/// `let orientation = (acc, gyro_rps).fuse_acc_gyro_using(&mut madgwick, dt);`
/// ```
/// use vqm::{Vector3f32,Quaternionf32};
/// use sensor_fusion::{MadgwickFilterf32,SensorFusion,FuseAccGyro};
///
/// let mut madgwick_filter = MadgwickFilterf32::default();
///
/// let delta_t: f32 = 0.0;
/// let acc = Vector3f32::default();
/// let gyro_rps = Vector3f32::default();
///
/// let orientation = (acc, gyro_rps).fuse_acc_gyro_using(&mut madgwick_filter, delta_t);
/// assert_eq!(orientation, Quaternionf32 { w: 1.0, x: 0.0, y: 0.0, z: 0.0 });
/// ```
pub trait FuseAccGyro<T> {
    fn fuse_acc_gyro_using<F: SensorFusion<T>>(self, filter: &mut F, delta_t: T) -> Quaternion<T>;
}

impl<T> FuseAccGyro<T> for (Vector3<T>, Vector3<T>) {
    fn fuse_acc_gyro_using<F: SensorFusion<T>>(self, filter: &mut F, delta_t: T) -> Quaternion<T> {
        let (acc, gyro) = self;
        filter.fuse_acc_gyro(acc, gyro, delta_t)
    }
}

#[allow(clippy::doc_paragraphs_missing_punctuation)]
/// Trait to allow sensor fusion filters to be used with method-call syntax, ie:<br>
/// `let orientation = (acc, gyro_rps).fuse_acc_gyro_using(&mut madgwick, dt);`
/// ```
/// use vqm::{Vector3f32,Quaternionf32};
/// use sensor_fusion::{MadgwickFilterf32,SensorFusion,FuseAccGyroMag};
/// let mut madgwick_filter = MadgwickFilterf32::default();
/// let dt: f32 = 0.0;
/// let acc = Vector3f32::default();
/// let gyro_rps = Vector3f32::default();
/// let mag = Vector3f32::default();
/// let orientation = (acc, gyro_rps, mag).fuse_acc_gyro_mag_using(&mut madgwick_filter, dt);
/// assert_eq!(orientation, Quaternionf32 { w: 1.0, x: 0.0, y: 0.0, z: 0.0 });
/// ```
pub trait FuseAccGyroMag<T> {
    fn fuse_acc_gyro_mag_using<F: SensorFusion<T>>(self, sensor_fusion_filter: &mut F, delta_t: T) -> Quaternion<T>;
}

impl<T> FuseAccGyroMag<T> for (Vector3<T>, Vector3<T>, Vector3<T>) {
    fn fuse_acc_gyro_mag_using<F: SensorFusion<T>>(self, sensor_fusion_filter: &mut F, delta_t: T) -> Quaternion<T> {
        let (acc, gyro, mag) = self;
        sensor_fusion_filter.fuse_acc_gyro_mag(acc, gyro, mag, delta_t)
    }
}

/*
/// Calculate quaternion derivative (dq/dt aka q_dot) from angular rate <https://ahrs.readthedocs.io/en/latest/filters/angular.html#quaternion-derivative>
pub fn q_dot<T>(q: Quaternion<T>, gyro_rps: Vector3<T>) -> Quaternion<T>
where
    T: Copy + One + Neg<Output = T> + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T>,
{
    let half = T::one() / (T::one() + T::one());
    Quaternion {
        w: (-q.x * gyro_rps.x - q.y * gyro_rps.y - q.z * gyro_rps.z) * half,
        x: (q.w * gyro_rps.x + q.y * gyro_rps.z - q.z * gyro_rps.y) * half,
        y: (q.w * gyro_rps.y - q.x * gyro_rps.z + q.z * gyro_rps.x) * half,
        z: (q.w * gyro_rps.z + q.x * gyro_rps.y - q.y * gyro_rps.x) * half,
    }
}*/

#[cfg(any(debug_assertions, test))]
mod tests {
    #![allow(clippy::wildcard_imports)]
    use super::*;
    use vqm::Vector3f32;

    #[allow(dead_code)]
    pub struct TestStruct;
    impl SensorFusion<f32> for TestStruct {
        fn set_gains(&mut self, _gain0: f32, _gain1: f32) {}
        fn gains(&self) -> (f32, f32) {
            (0.0, 0.0)
        }

        fn requires_initialization() -> bool {
            true
        }
        fn fuse_acc_gyro(&mut self, _acc: Vector3f32, _gyro_rps: Vector3f32, _delta_t: f32) -> Quaternionf32 {
            Quaternionf32::default()
        }
        fn fuse_acc_gyro_mag(
            &mut self,
            acc: Vector3f32,
            gyro_rps: Vector3f32,
            _mag: Vector3f32,
            delta_t: f32,
        ) -> Quaternionf32 {
            self.fuse_acc_gyro(acc, gyro_rps, delta_t)
        }
        fn correct_yaw_with_gain(&mut self, _yaw_radians: f32, _gain: f32, _delta_t: f32) -> Quaternionf32 {
            Quaternionf32::default()
        }
    }

    //#[allow(dead_code)]
    #[test]
    fn sensor_fusion() {
        let mut test_struct: TestStruct = TestStruct {};
        _ = TestStruct::requires_initialization();
        //assert_eq!(TestStruct::requires_initialization(), true);

        test_struct.set_gains(0.0, 0.0);

        let delta_t: f32 = 0.0;
        let acc = Vector3f32::default();
        let gyro_rps = Vector3f32::default();

        let orientation = test_struct.fuse_acc_gyro(acc, gyro_rps, delta_t);
        assert_eq!(orientation, Quaternion::default());
    }

    #[test]
    fn fuse_using() {
        use crate::MadgwickFilterf32;

        let mut madgwick_filter = MadgwickFilterf32::default();
        let requires_initialization = MadgwickFilterf32::requires_initialization();
        assert!(requires_initialization);

        madgwick_filter.set_beta(1.0);

        let delta_t: f32 = 0.0;
        let acc = Vector3f32::default();
        let gyro_rps = Vector3f32::default();

        //let orientation = madgwick_filter.fuse_acc_gyro(acc, gyro_rps, delta_t);
        let orientation = (acc, gyro_rps).fuse_acc_gyro_using(&mut madgwick_filter, delta_t);
        assert_eq!(orientation, Quaternion { w: 1.0, x: 0.0, y: 0.0, z: 0.0 });
    }
}
