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
mod madgwick_filter;
mod mahony_filter;

mod kalman_filter_z;
mod kalman_filter_z_with_r;

mod kalman_filter_xyz;
mod kalman_filter_xyz_with_r;

mod sensor_fusion;
mod sensor_fusion_math;

mod trilaterate_2d;
mod trilaterate_3d;

pub use complementary_filter::{ComplementaryFilter, ComplementaryFilterf32, ComplementaryFilterf64};
pub use madgwick_filter::{MadgwickFilter, MadgwickFilterf32, MadgwickFilterf64};
pub use mahony_filter::{MahonyFilter, MahonyFilterf32, MahonyFilterf64};

pub use kalman_filter_z::{KalmanFilterZ, KalmanFilterZf32, KalmanFilterZf64};
pub use kalman_filter_z_with_r::{KalmanFilterZWithR, KalmanFilterZWithRf32, KalmanFilterZWithRf64};

pub use kalman_filter_xyz::{KalmanFilterXYZ, KalmanFilterXYZf32};
pub use kalman_filter_xyz_with_r::{KalmanFilterXYZWithR, KalmanFilterXYZWithRf32};

pub use sensor_fusion::{FuseAccGyro, FuseAccGyroMag, SensorFusion};
pub use sensor_fusion_math::SensorFusionMath;

pub use trilaterate_2d::{Anchor2d, Anchor2df32, Anchor2df64, trilaterate_2d};
pub use trilaterate_3d::{Anchor3d, Anchor3df32, Anchor3df64, trilaterate_3d_weighted};
