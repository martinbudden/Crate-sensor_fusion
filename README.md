# `sensor-fusion` Rust Crate ![license](https://img.shields.io/badge/license-MIT-green) [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) ![open source](https://badgen.net/badge/open/source/blue?icon=github)

This crate contains [sensor fusion](https://en.wikipedia.org/wiki/Sensor_fusion) algorithms to combine
output from a gyroscope, accelerometer, and optionally a magnetometer to give output that has less uncertainty
than the output of the individual sensors.

Six sensor fusion implementations are available:

1. Complementary Filter
2. Mahony Filter
3. Madgwick Filter
4. Altitude Kalman Filter
5. 2D Position Kalman Filter.
6. 3D Position Kalman Filter.

## Simple example

Here's a simple example that uses a Madgwick filter to calculate the orientation by fusing accelerometer and gyro values:

```rust
use sensor_fusion::{MadgwickFilterf32, SensorFusion};
use vqm::Vector3f32;

fn main() {
    let mut madgwick = MadgwickFilterf32::new();
    let dt = 0.001; // 1 millisecond

    // Mock sensor values, gyro converted to rps (values normally read from IMU).
    let gyro_rps = Vector3f32::new(90.0, 8.0, 10.0).to_radians();
    let acc = Vector3f32::new(0.1, 0.2, 0.9);

    // Fuse acc and gyro values
    let orientation = madgwick.fuse_acc_gyro(acc, gyro_rps, dt);

    let (roll, pitch, yaw) = orientation.calculate_euler_angles_degrees();

    // Print out the Euler angles.
    println!("pitch={}, roll={}, yaw={}", pitch, roll, yaw);
}
```

## Madgwick Filter

The Madgwick filter has been refactored to be more computationally efficient (and so faster) than
the standard version used in many implementations, see [Optimization](#opt) below.

The Madgwick filter also supports three-way fusing of accelerometer, gyroscope, and magnetometer readings:

```text
let orientation = madgwick.fuse_acc_gyro_mag(acc, gyro_rps, mag, dt);
```

## Mahony filter

The Mahony filter has the same interface as the Madgwick filter:

```text
use crate::sensor_fusion::{MahonyFilterf32,SensorFusion};

let mut mahony = MahonyFilterf32::default();

let orientation = mahony.fuse_acc_gyro(acc, gyro_rps, dt);
```

The Mahony filter does not support 3-way fusion using a magnetometer.

## Complementary filter

This has been implemented for reference. It is not recommended for general use:
both the Mahony filter and the Madgwick filter are faster and don't suffer from gimbal lock.

## Method call interface

The `FuseAccGyro` and `FuseAccGyroMag` traits allow method-call syntax to be used:

```rust
use {crate::sensor_fusion::{FuseAccGyro, FuseAccGyroMag, MadgwickFilterf32}, vqm::Vector3f32};

let dt = 0.001;
let mut madgwick = MadgwickFilterf32::default();
let acc = Vector3f32::default();
let gyro = Vector3f32::default();
let mag = Vector3f32::default();

let orientation = (acc, gyro).fuse_acc_gyro_using(&mut madgwick, dt);
// or
let orientation = (acc, gyro, mag).fuse_acc_gyro_mag_using(&mut madgwick, dt);
```

## Kalman filter

Three Kalman filters are provided:

1. `KalmanFilterZ` - a filter to estimate altitude and vertical speed.
2. `KalmanFilterXY` - a filter to estimate 2D position and velocity.
3. `KalmanFilterXYZ` - a filter to estimate 3D position and velocity.

## Position Kalman filter internals

A Kalman filter works by predicting an object's state using a physical model of the object. So, for example,
to predict an object's position velocity it uses the measured acceleration and the kinematic equations
`s = u*t + 0.5*a*t^2` and `v = u + a*t`.

When it obtains a reading for the object's actual position (for example from a GPS) it uses that reading
to correct the prediction (essentially by taking a weighted average of the predicted state and the reading).

The naive implementation of a 3D Position Kalman filter is extremely computationally intensive.

To predict 3D position using a gyroscope, accelerometer, you need a massive 15x15 covariance matrix, even in 2D
you need a 10x10 matrix.

Furthermore, since combining gyroscope and accelerometer measurements is an non-linear operation, a Linear Kalman Filter (LKF),
cannot be used: an Extended Kalman Filter (EKF) is required. An EKF is even more computationally intensive:
as well as doing the multiplication of these large matrices it also needs to calculate
[Jacobians](https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant).

Fortunately there are a number of things that can be done to reduce this computational load.

Firstly we remove the responsibility for calculating the orientation from the Kalman filter:
we use a another sensor fusion filter (ie a Mahony or Madgwick) filter to do this.

This reduces the covariance matrix to more manageable size of 9x9.
What's more, prediction using a accelerometer alone is a linear operation, so we can use a LFK rather than an EKF.

```text
   ┌─────┐  Acc/Gyro  ┌─────────────────┐
   │ IMU ├──────────► │ MADGWICK FILTER ├──► Orientation Quaternion
   └─────┘            └─────────────────┘
      │
      │ Acc
      │            ┌────────────────────────┐
      └──────────► │ POSITION KALMAN FILTER ├──► Position Vector
                   └────────────────────────┘    Velocity Vector
                                │
   GPS & Barometer  ──►─────────┘
```

But we are still not out of the woods: multiplying two 9x9 matrices requires 729 individual arithmetic operations,
matrix inversion even more.
And the Kalman filter predict and correct steps each require several matrix operations.

TODO: further explanation of Kalman filter internals in readme.

## SIMD support

**SIMD** support (for the `f32` variants) can be enabled with the `simd` feature.

It is currently experimental, so if you used SIMD make sure you benchmark to show that you are indeed getting
a performance improvement over the non-SIMD version.

This uses [portable simd](https://doc.rust-lang.org/core/simd/index.html), which requires the nightly compiler, since it is still
unstable in rust.

This can be invoked using `rustup`, eg:

```sh
rustup run nightly cargo build --features simd --target thumbv8m.main-none-eabi
```

## Madgwick Filter Optimization {#opt}

Classically, the calculation of the Madgwick gradient descent corrective step involves multiplication of a vector by a matrix,
this involves a total of 54 arithmetic operations for the acc/gyro case.

However, because both the matrix and vector contain zero elements, and because there is some symmetry in the matrix,
the calculation can be refactored to use fewer arithmetic operations.

Indeed it can be reduced to a total of 31 arithmetic operations.
By using SIMD this can be further reduced to 16 operations (11 scalar and 5 SIMD operations). See below.

I haven't yet benchmarked this Rust implementation,
but the original C++ version of `MadgwickFilter::update_orientation` (equivalent to `madgwick.fuse_acc_gyro`)
took or under 20 microseconds on a 240 MHz ESP32 S3.

The aim is to be able to run sensor fusion a part of a Gyro/PID loop running at 8kHz. That means everything
(including reading the IMU, filtering the output, performing sensor fusion and calculation the motor outputs
using a PID controller) needs to run in 125 microseconds. This is currently looking achievable.

```rust
// Classic version
//
// total:
//      54 arithmetic operations (35 multiplications, 19 additions/subtractions)
//
# use vqm::{Quaternionf32, Vector3f32, Vector4f32, Matrix4x4f32};
fn madgwick_step(q: Quaternionf32, a: Vector3f32) -> Vector4f32 {
    let M = Matrix4x4f32::new([ // 10 multiplications
        -2.0*q.x, 2.0*q.w,      0.0, 0.0,
         2.0*q.y, 2.0*q.z, -4.0*q.w, 0.0,
        -2.0*q.z, 2.0*q.y, -4.0*q.x, 0.0,
         2.0*q.w, 2.0*q.x,      0.0, 0.0
    ]);

    let v = Vector4f32::new( // 9 multiplications, 7 additions/subtractions
        2.0*(      q.w*q.y - q.z*q.x) - a.x,
        2.0*(      q.z*q.w + q.x*q.y) - a.y,
        2.0*(0.5 - q.w*q.w - q.x*q.x) - a.z,
        0.0
    );

    M * v // 16 multiplications, 12 additions
}
```

```rust
// Refactored version
//
// total:
//      31 arithmetic operations (19 multiplications, 12 additions/subtractions)
// when converted to SIMD this becomes:
//      16 operations (7 multiplications, 4 additions, 3 vector multiplications, 2 vector additions)
//
# use vqm::{Quaternionf32, Vector3f32};
fn madgwick_step(q: Quaternionf32, a: Vector3f32) -> Quaternionf32 {
    let wz_common = 2.0 * (q.x * q.x + q.y * q.y); // 3 multiplications, 1 addition
    let xy_common = 2.0 * (q.w * q.w + q.z * q.z - 1.0 + 2.0 * wz_common + a.z); // 4 multiplications, 3 additions/subtractions

    Quaternionf32 { // 12 multiplications, 8 additions/subtractions
        w: q.w * wz_common + q.y * a.x - q.x * a.y,
        x: q.x * xy_common - q.z * a.x - q.w * a.y,
        y: q.y * xy_common + q.w * a.x - q.z * a.y,
        z: q.z * wz_common - q.x * a.x - q.y * a.y,
    }
}
```

## Original implementation

I originally implemented this crate as a C++ library:
[Library-SensorFusion](https://github.com/martinbudden/Library-SensorFusion).

## License

Licensed under either of:

* Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0)>
* MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT)>

at your option.
