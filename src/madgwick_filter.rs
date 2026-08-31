use crate::{SensorFusion, SensorFusionMath};
use num_traits::{ConstOne, ConstZero, float::FloatCore};
use vqm::{MathMethods, Quaternion, QuaternionMath, Vector3};

/// Madgwick filter for `f32`<br>
pub type MadgwickFilterf32 = MadgwickFilter<f32>;
/// Madgwick filter for `f64`<br><br>
pub type MadgwickFilterf64 = MadgwickFilter<f64>;

pub trait ConstFour {
    const FOUR: Self;
}
impl ConstFour for f32 {
    const FOUR: Self = 4.0;
}
impl ConstFour for f64 {
    const FOUR: Self = 4.0;
}

/// [Madgwick filter](https://ahrs.readthedocs.io/en/latest/filters/madgwick.html).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MadgwickFilter<T> {
    /// orientation quaternion.
    q: Quaternion<T>,
    max_acc_magnitude_squared: T,
    /// Beta gain tuned for fast tracking of pitch and roll (typically 0.01 to 0.1).
    beta: T,
    /// Beta yaw gain, should be significantly smaller than beta (e.g., 0.001 to 0.005).
    /// GPS latency can cause the filter to over correct and oscillate if this value is too aggressive.
    beta_yaw: T,
}

impl<T> Default for MadgwickFilter<T>
where
    T: Copy + ConstZero + ConstOne + ConstFour,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> MadgwickFilter<T>
where
    T: Copy + ConstZero + ConstOne + ConstFour,
{
    /// Constructor.
    pub const fn with_orientation_and_beta(orientation: Quaternion<T>, beta: T) -> Self {
        MadgwickFilter { q: orientation, max_acc_magnitude_squared: T::FOUR, beta, beta_yaw: T::ZERO }
    }

    /// Constructor.
    pub const fn with_orientation(orientation: Quaternion<T>) -> Self {
        MadgwickFilter { q: orientation, max_acc_magnitude_squared: T::FOUR, beta: T::ONE, beta_yaw: T::ZERO }
    }

    /// Constructor.
    #[must_use]
    pub const fn new() -> Self {
        MadgwickFilter {
            q: Quaternion { w: T::ONE, x: T::ZERO, y: T::ZERO, z: T::ZERO },
            max_acc_magnitude_squared: T::FOUR,
            beta: T::ONE,
            beta_yaw: T::ZERO,
        }
    }
}

impl<T: Copy> MadgwickFilter<T> {
    pub fn set_orientation(&mut self, orientation: Quaternion<T>) {
        self.q = orientation;
    }
    pub fn set_beta(&mut self, beta: T) {
        self.beta = beta;
    }
    pub fn set_beta_yaw(&mut self, beta_yaw: T) {
        self.beta_yaw = beta_yaw;
    }
}
impl<T> MadgwickFilter<T>
where
    T: Copy + FloatCore + QuaternionMath + SensorFusionMath + MathMethods,
{
    pub fn correct_yaw(&mut self, yaw_radians: T, delta_t: T) -> Quaternion<T> {
        self.correct_yaw_with_gain(yaw_radians, self.beta_yaw, delta_t)
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
/// and also x-io Technologies [sensor fusion library](https://github.com/xioTechnologies/Fusion).
///
/// For computation efficiency this code refactors the code used in many implementations (Arduino, Adafruit, M5 Stack, Reefwing-AHRS),
/// [see Madgwick refactoring](../../../documents/MadgwickRefactoring.md).
///
impl<T> SensorFusion<T> for MadgwickFilter<T>
where
    T: Copy + FloatCore + QuaternionMath + SensorFusionMath + MathMethods,
{
    fn set_gains(&mut self, gain0: T, gain1: T) {
        self.beta = gain0;
        self.beta_yaw = gain1;
    }
    fn gains(&self) -> (T, T) {
        (self.beta, self.beta_yaw)
    }

    fn requires_initialization() -> bool {
        true
    }

    /// Fuses accelerometer and gyroscope readings to give the orientation quaternion.
    fn fuse_acc_gyro(&mut self, acc: Vector3<T>, gyro_rps: Vector3<T>, delta_t: T) -> Quaternion<T> {
        // Calculate the corrective step.
        let step = SensorFusionMath::madgwick_step_acc(self.q, acc, self.max_acc_magnitude_squared);

        // Calculate quaternion derivative (q_dot, aka dq/dt) from the angular rate and subtract the corrective step.
        let q_dot = SensorFusionMath::derivative(self.q, gyro_rps) - step * self.beta;

        // Update the orientation quaternion using simple Euler integration
        self.q += q_dot * delta_t;

        // normalize the orientation quaternion and return it
        self.q = self.q.normalize();
        self.q
    }

    /// Fuses accelerometer, gyroscope, and magnetometer readings to give the orientation quaternion.
    fn fuse_acc_gyro_mag(
        &mut self,
        acc: Vector3<T>,
        gyro_rps: Vector3<T>,
        mag: Vector3<T>,
        delta_t: T,
    ) -> Quaternion<T> {
        // Calculate the corrective step.
        let step = SensorFusionMath::madgwick_step_acc_mag(self.q, acc, mag, self.max_acc_magnitude_squared);

        // Calculate quaternion derivative (q_dot, aka dq/dt) from the angular rate and subtract the corrective step.
        let q_dot = SensorFusionMath::derivative(self.q, gyro_rps) - step * self.beta;

        // Update the orientation quaternion using simple Euler integration
        self.q += q_dot * delta_t;

        // normalize the orientation quaternion and return it
        self.q = self.q.normalize();
        self.q
    }

    /// Note `delta_t` is the time since `correct_yaw` was last called, distinct from the `delta_t` in `fuse_acc_gyro`.
    /// This function called at a frequency of ~10Hz if called from a GPS update, or ~200Hz for a magnetometer update.
    /// It is normally called withing the Gyro/PID loop and so should be as fast as possible to minimize jitter.
    fn correct_yaw_with_gain(&mut self, yaw_radians: T, gain: T, delta_t: T) -> Quaternion<T> {
        // Create a target yaw quaternion from the yaw heading.
        // This represents a flat orientation pointing in the direction of travel.
        let mut target = Quaternion::<T>::from_yaw_radians(yaw_radians);

        // The dot product between your two quaternions tells you how well aligned they are.
        // If the dot product is positive, the two quaternions are on the same hemisphere, and moving directly toward the target is the shortest path.
        // If the dot product is negative, the quaternions are pointing the same way geometrically,
        // but traveling along the raw path would take the "long way around" the 4D sphere
        // (due to the double-cover property of quaternions where `q` and `-q` represent the same 3D rotation).
        // If the dot product is negative, we simply negate the target quaternion.
        // This flips it to the closer hemisphere, ensuring the gradient descent step takes the shortest possible path.
        if self.q.dot(target) < T::zero() {
            target = -target;
        }

        // Calculate the error quaternion between current orientation and the target (measured) yaw heading
        let error = self.q * target.conjugate();

        // Construct a pure rotational velocity quaternion around the Z-axis
        let omega = Quaternion::<T>::new(T::zero(), T::zero(), T::zero(), -error.z);

        // Calculate kinematic derivative: q_dot = 0.5 * q * omega
        // The 0.5 factor is absorbed into the `beta_yaw` value, so does not appear here.
        let q_dot = self.q * omega;

        //let error_z = -self.q.w * target.z - self.q.x * target.y + self.q.y * target.x + self.q.z * target.w;
        //let q_dot = Quaternion { w: self.q.z, x: -self.q.y, y: self.q.x, z: -self.q.w } * (error_z * self.beta_yaw);

        // Euler integration & renormalization
        self.q += q_dot * (delta_t * gain);
        self.q = self.q.normalize();

        self.q
    }
}

#[cfg(test)]
mod test_traits {
    use super::*;

    fn is_full<T: Sized + Send + Sync + Unpin + Copy + Clone + Default + PartialEq>() {}

    #[test]
    fn normal_types() {
        is_full::<MadgwickFilter<f32>>();
    }
}
#[cfg(any(debug_assertions, test))]
mod tests {
    #![allow(unused)]
    use super::*;

    use crate::FuseAccGyro;
    use vqm::{Quaternionf32, Vector3f32};

    #[test]
    #[allow(clippy::float_cmp)]
    fn q_dot() {
        let q = Quaternionf32::new(2.0, 3.0, 5.0, 7.0);
        let mut target = Quaternionf32::new(11.0, 13.0, 17.0, 19.0);
        if q.dot(target) < 0.0 {
            target = -target;
        }

        let error = q * target.conjugate();
        assert_eq!(53.0, error.z);
        let omega = Quaternionf32::new(0.0, 0.0, 0.0, -error.z);
        let q_dot = q * omega;
        assert_eq!(Quaternionf32 { w: 371.0, x: -265.0, y: 159.0, z: -106.0 }, q_dot);

        let error_z = -q.w * target.z - q.x * target.y + q.y * target.x + q.z * target.w;
        assert_eq!(53.0, error_z);
        assert_eq!(error.z, error_z);
        let q_dot2 = Quaternion { w: q.z * error_z, x: -q.y * error_z, y: q.x * error_z, z: -q.w * error_z };
        assert_eq!(q_dot, q_dot2);
    }

    #[test]
    fn readme() {
        let mut madgwick_filter = MadgwickFilterf32::default();

        let dt: f32 = 0.001;
        let acc = Vector3f32::default();
        let gyro_rps = Vector3f32::default();

        let orientation = madgwick_filter.fuse_acc_gyro(acc, gyro_rps, dt);
        assert_eq!(orientation, Quaternion { w: 1.0, x: 0.0, y: 0.0, z: 0.0 });
    }

    #[test]
    fn update_orientation() {
        let mut madgwick_filter = MadgwickFilterf32::default();
        let requires_initialization = MadgwickFilterf32::requires_initialization();
        assert!(requires_initialization);

        madgwick_filter.set_beta(1.0);

        let delta_t: f32 = 0.0;
        let acc = Vector3f32::default();
        let gyro_rps = Vector3f32::default();

        let orientation = madgwick_filter.fuse_acc_gyro(acc, gyro_rps, delta_t);
        assert_eq!(orientation, Quaternion { w: 1.0, x: 0.0, y: 0.0, z: 0.0 });
    }

    #[test]
    fn fuse_acc_gyro_using() {
        let mut madgwick_filter = MadgwickFilterf32::default();

        let delta_t: f32 = 0.0;
        let acc = Vector3f32::default();
        let gyro_rps = Vector3f32::default();

        let orientation = (acc, gyro_rps).fuse_acc_gyro_using(&mut madgwick_filter, delta_t);
        assert_eq!(orientation, Quaternion { w: 1.0, x: 0.0, y: 0.0, z: 0.0 });
    }

    #[allow(clippy::float_cmp)]
    #[test]
    fn yaw_correction() {
        let initial_yaw = 179.0f32;
        let initial_q = Quaternionf32::from_yaw_degrees(initial_yaw);
        //println!("Initial q: {initial_q}");

        let mut filter = MadgwickFilterf32::with_orientation(initial_q);
        let gain = 5.0;
        let delta_t = 0.1;
        filter.set_beta_yaw(gain);

        let target_yaw = 178.0_f32;
        let filtered_q = filter.correct_yaw_with_gain(target_yaw.to_radians(), gain, delta_t);
        let final_yaw = filtered_q.calculate_yaw_degrees();
        println!("Yaw, initial: {initial_yaw:.4}° | target: {target_yaw:.4}° | filtered: {final_yaw:.4}°");
        assert!((178.5 - final_yaw).abs() < 2e-5);

        filter.set_orientation(initial_q);
        let target_yaw = 179.5_f32;
        let filtered_q = filter.correct_yaw_with_gain(target_yaw.to_radians(), gain, delta_t);
        let final_yaw = filtered_q.calculate_yaw_degrees();
        println!("Yaw, initial: {initial_yaw:.4}° | target: {target_yaw:.4}° | filtered: {final_yaw:.4}°");
        assert!((179.25 - final_yaw).abs() < 2e-5);

        let initial_yaw = 179.75f32;
        let initial_q = Quaternionf32::from_yaw_degrees(initial_yaw);
        filter.set_orientation(initial_q);
        let target_yaw = 181.0_f32;
        let filtered_q = filter.correct_yaw_with_gain(target_yaw.to_radians(), gain, delta_t);
        let final_yaw = filtered_q.calculate_yaw_degrees();
        println!("Yaw, initial: {initial_yaw:.4}° | target: {target_yaw:.4}° | filtered: {final_yaw:.4}°");
        assert!((-179.625 - final_yaw).abs() < 4e-5);
        // A correct step moves across the 180° threshold. Because of how atan2 wraps,
        // moving from 179.75° towards 181.0° means it wraps cleanly into the negative space (e.g. -179.625°).
        // If it took the wrong long path, the angle would drop below 179.75° (e.g. 179.25°).
        assert!(
            !(0.0..=179.0).contains(&final_yaw),
            "\n**** Filter took the long way around! Yaw dropped down to {final_yaw:.2}°"
        );
    }
}
