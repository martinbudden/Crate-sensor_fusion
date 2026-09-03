use num_traits::{ConstOne, ConstZero, float::FloatCore};
use vqm::{MathMethods, Matrix3x3Math, Vector3};

use super::{KalmanFilterZ, kalman_filter_z::KalmanFilterZConstants};

/// `f32` variant of `AltitudeKalmanFilter`.
pub type KalmanFilterZWithRf32 = KalmanFilterZWithR<f32>;
/// `f64` variant of `AltitudeKalmanFilter`.
pub type KalmanFilterZWithRf64 = KalmanFilterZWithR<f64>;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KalmanFilterZWithR<T> {
    pub base: KalmanFilterZ<T>,
    /// Barometer measurement variance.
    r_barometer: T,
    /// Rangefinder measurement variance.
    r_rangefinder: T,
    /// GPS measurement variance.
    r_gps: T,
}

impl<T> Default for KalmanFilterZWithR<T>
where
    T: Copy + ConstZero + ConstOne + FloatCore + Matrix3x3Math + KalmanFilterZConstants,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> KalmanFilterZWithR<T>
where
    T: Copy + ConstZero + ConstOne + FloatCore + Matrix3x3Math + KalmanFilterZConstants,
{
    /// Constructor.
    #[must_use]
    pub const fn new() -> Self {
        Self { base: KalmanFilterZ::new(), r_barometer: T::ZERO, r_rangefinder: T::ZERO, r_gps: T::ZERO }
    }
}

impl<T> KalmanFilterZWithR<T>
where
    T: Copy + ConstZero + ConstOne + FloatCore + MathMethods + Matrix3x3Math,
{
    pub fn predict(&mut self, acceleration_measurement: T, delta_t: T) -> Vector3<T> {
        self.base.predict(acceleration_measurement, delta_t)
    }
}

impl<T> KalmanFilterZWithR<T>
where
    T: Copy + ConstZero + ConstOne + FloatCore + Matrix3x3Math + KalmanFilterZConstants,
{
    /// Phase 2: Correct altitude using the barometer measurement.
    #[inline]
    pub fn correct_altitude_using_barometer(&mut self, altitude: T) {
        self.base.correct_altitude(altitude, self.r_barometer);
    }

    /// Phase 2: Correct altitude using the rangefinder measurement.
    #[inline]
    pub fn correct_altitude_using_rangefinder(&mut self, altitude: T) {
        self.base.correct_altitude(altitude, self.r_barometer);
    }
    /// Phase 2: Correct altitude using GPS vertical measurement.
    #[inline]
    pub fn correct_altitude_using_gps(&mut self, altitude: T) {
        self.base.correct_altitude(altitude, self.r_gps);
    }
}

#[cfg(test)]
mod test_traits {
    use super::*;

    fn is_full<T: Sized + Send + Sync + Unpin + Copy + Clone + Default + PartialEq>() {}

    #[test]
    fn normal_types() {
        is_full::<KalmanFilterZWithRf32>();
        is_full::<KalmanFilterZWithRf64>();
    }
}
