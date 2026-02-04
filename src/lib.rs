#![no_std]
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]
#![deny(unused_must_use)]

mod complementary_filter;
mod madgwick_filter;
mod mahony_filter;
mod sensor_fusion;

pub use sensor_fusion::SensorFusion;
pub use sensor_fusion::q_dot;
pub use complementary_filter::ComplementaryFilter;
pub use madgwick_filter::MadgwickFilter;
pub use mahony_filter::MahonyFilter;
