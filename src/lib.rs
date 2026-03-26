#![doc = include_str!("../README.md")]

#![no_std]
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]
#![deny(unused_must_use)]

mod complementary_filter;
mod madgwick_filter;
mod mahony_filter;
mod sensor_fusion;

pub use complementary_filter::{ComplementaryFilter, ComplementaryFilterf32, ComplementaryFilterf64};
pub use madgwick_filter::{MadgwickFilter, MadgwickFilterf32, MadgwickFilterf64};
pub use mahony_filter::{MahonyFilter, MahonyFilterf32, MahonyFilterf64};
pub use sensor_fusion::SensorFusion;
pub use sensor_fusion::q_dot;
