#![cfg_attr(feature = "simd", feature(portable_simd))]
#![doc = include_str!("../README.md")]
#![cfg_attr(not(test), no_std)]
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]
//#![deny(missing_docs)]
#![deny(
    missing_copy_implementations,
    missing_debug_implementations,
    trivial_casts,
    trivial_numeric_casts,
    unused_must_use,
    unused_extern_crates,
    unused_import_braces,
    unused_qualifications,
    unused_results
)]
#![warn(unused_results)]
#![warn(clippy::pedantic)]
#![warn(clippy::doc_paragraphs_missing_punctuation)]

mod complementary_filter;
mod kalman_state_vector9;
mod madgwick_filter;
mod mahony_filter;

mod altitude_kalman_filter;
mod altitude_kalman_filter_with_r;

mod position_kalman_filter;
mod position_kalman_filter_with_r;

mod sensor_fusion;
mod sensor_fusion_math;

mod trilaterate_2d;
mod trilaterate_3d;

pub use complementary_filter::{ComplementaryFilter, ComplementaryFilterf32, ComplementaryFilterf64};
pub use madgwick_filter::{MadgwickFilter, MadgwickFilterf32, MadgwickFilterf64};
pub use mahony_filter::{MahonyFilter, MahonyFilterf32, MahonyFilterf64};

pub use altitude_kalman_filter::{AltitudeKalmanFilter, AltitudeKalmanFilterf32, AltitudeKalmanFilterf64};
pub use altitude_kalman_filter_with_r::{
    AltitudeKalmanFilterWithR, AltitudeKalmanFilterWithRf32, AltitudeKalmanFilterWithRf64,
};

pub use kalman_state_vector9::{KalmanStateVector9, KalmanStateVector9f32, KalmanStateVector9f64};
pub use position_kalman_filter::{PositionKalmanFilter, PositionKalmanFilterf32};
pub use position_kalman_filter_with_r::{PositionKalmanFilterWithR, PositionKalmanFilterWithRf32};

pub use sensor_fusion::{FuseAccGyro, FuseAccGyroMag, SensorFusion};
pub use sensor_fusion_math::SensorFusionMath;

pub use trilaterate_2d::{Anchor2d, Anchor2df32, Anchor2df64, trilaterate_2d};
pub use trilaterate_3d::{Anchor3d, Anchor3df32, Anchor3df64, trilaterate_3d_weighted};
