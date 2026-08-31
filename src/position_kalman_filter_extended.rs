use vqm::Vector3f32;

use super::PositionKalmanFilter;

/// `f32` variant of `PositionKalmanFilterExtended`.
pub type PositionKalmanFilterExtendedf32 = PositionKalmanFilterExtended;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PositionKalmanFilterExtended {
    pub base: PositionKalmanFilter,
    /// Absolute Measurement Noise variance for horizontal GPS channels.
    pub r_gps_horizontal: f32,
    /// Absolute Measurement Noise variance for vertical GPS channels.
    pub r_gps_vertical: f32,
    /// Absolute Measurement Noise variance for barometric pressure altimeter.
    pub r_barometer: f32,
    /// Absolute Measurement Noise variance for rangefinder.
    pub r_rangefinder: f32,
    /// Absolute Measurement Noise variance for optical flow.
    pub r_optical_flow: Vector3f32,
}

impl Default for PositionKalmanFilterExtended {
    fn default() -> Self {
        Self::new()
    }
}

impl PositionKalmanFilterExtended {
    /// Constructor.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            base: PositionKalmanFilter::new(),
            r_gps_horizontal: 0.0,
            r_gps_vertical: 0.0,
            r_barometer: 0.0,
            r_rangefinder: 0.0,
            r_optical_flow: Vector3f32 { x: 0.0, y: 0.0, z: 0.0 },
        }
    }
}

impl PositionKalmanFilterExtended {
    pub fn predict_states(&mut self, acc_measurement: Vector3f32, dt: f32) {
        self.base.predict_states(acc_measurement, dt);
    }

    pub fn predict_covariance(&mut self, dt: f32) {
        self.base.predict_covariance(dt);
    }

    /// Phase 2: Correct altitude using the barometer measurement.
    #[inline]
    pub fn correct_altitude_using_barometer(&mut self, altitude: f32) {
        self.base.correct_altitude(altitude, self.r_barometer);
    }

    /// Phase 2: Correct altitude using the rangefinder measurement.
    #[inline]
    pub fn correct_altitude_using_rangefinder(&mut self, altitude: f32) {
        self.base.correct_altitude(altitude, self.r_rangefinder);
    }

    /// Phase 2: Correct altitude using GPS vertical measurement.
    #[inline]
    pub fn correct_altitude_using_gps(&mut self, altitude: f32) {
        self.base.correct_altitude(altitude, self.r_gps_vertical);
    }
    /// Phase 2: Correct position using GPS position measurement (typically at a 1Hz to 10Hz rate).
    pub fn correct_position_using_gps(&mut self, position: Vector3f32) {
        let r_gps = Vector3f32 { x: self.r_gps_horizontal, y: self.r_gps_horizontal, z: self.r_gps_vertical };
        self.base.correct_position(position, r_gps);
    }

    /// Phase 2: Correct position using optical flow position measurement.
    pub fn correct_position_using_optical_flow(&mut self, position: Vector3f32) {
        self.base.correct_position(position, self.r_optical_flow);
    }
}

#[cfg(test)]
mod test_traits {
    use super::*;

    fn is_full<T: Sized + Send + Sync + Unpin + Copy + Clone + Default + PartialEq>() {}

    #[test]
    fn normal_types() {
        is_full::<PositionKalmanFilterExtended>();
        is_full::<PositionKalmanFilterExtendedf32>();
    }
}
