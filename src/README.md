# Kinematic State-Space Sensor Fusion

This crate implements a high-performance, asynchronous, 9-state 3D Position Kalman Filter
optimized for embedded systems (`no_std`). It targets hardware platforms like the RP2040 and
RP2350 (Cortex-M33).

## 1. The State Vector (x)

The filter tracks a 9-dimensional kinematic state vector split into three 3D spatial domains:

```text
         ⎡ Position  (x, y, z) ⎤
     x = ⎢ Velocity  (x, y, z) ⎥
         ⎣ IMU Bias  (x, y, z) ⎦
```

In code, this mapping is managed via zero-cost abstractions by destructuring `Vector3f32` structures.

## 2. System Architecture: Asynchronous Multi-Sensor Execution Loop

In flight controller firmware, sensors execute at different rates (asynchronous execution).
This filter splits the prediction and correction cycles into modular functions:

```text
┌────────────────────────────────────────────────────────────────┐
│  IMU INTERRUPT TIMELOOP (~1000 Hz)                             │
│  1. Read Accel ──► predict_states()      [State Propagation]   │
│  2. Execute    ──► predict_covariance()  [P = A * E * Aᵀ + Q]  │
└────────────────────────────────────────────────────────────────┘
                            │
                            ▼
    Asynchronous Data Packets Available?
    ├── YES (Baro SPI @ ~40Hz)  ──► update_barometer(z)  [S = P₂₂ + R]
    └── YES (GPS UART @ ~5Hz)   ──► update_gps(x, y, z)  [S = H*P*Hᵀ + R]
```

## 3. Architectural Rule: The Accelerometer is NOT a Measurement Update

A common point of confusion is searching for an `update_accelerometer` function.
In a kinematic navigation filter, the accelerometer acts as the **Control Input (u)**
during the time-propagation step, pushing the physics model forward. Absolute reference sensors
(**GPS, Barometer**) act as **Measurements (z)** to strip away the integration drift.

## 4. Matrix Transformations

### Covariance Time Propagation: `P = A * E * Aᵀ + Q`

```text
    ⎡  1    0   -dT    ⎤          ⎡ -dT² * q_vel    0            0      ⎤
A = ⎢ dT    1     0    ⎥      Q = ⎢      0          0            0      ⎥
    ⎣  0    0   1+β*dT ⎦          ⎣      0          0      dT² * q_bias ⎦
```

### Multi-Dimensional GPS Innovation Matrix: `S = H * P * Hᵀ + R`

```text
                       ⎡ P₁₁  P₁₂  P₁₃ ⎤   ⎡ R_horiz     0         0     ⎤
S = (H * P * Hᵀ) + R = ⎢ P₂₁  P₂₂  P₂₃ ⎥ + ⎢    0     R_horiz      0     ⎥
                       ⎣ P₃₁  P₃₂  P₃₃ ⎦   ⎣    0        0      R_vert   ⎦
```

---

## 🛠️ API & Core Operator Mechanics

To achieve maximum speed on microcontrollers, this crate enforces four software design patterns:

1. **Pass-by-Value API:** Data structures arrive via registers (`FPU s0-s31`), bypassing stack pointer overhead.
2. **Fixed-Width Logical Padding:** Array dimensions are logically treated as 4-wide rows (`align(16)`) to ensure
LLVM auto-vectorization emits single-cycle parallel vector hardware instructions (`vmul.f32`).
3. **Analytic Inversion:** Multi-dimensional GPS steps leverage a zero-loop 3x3 determinant inverse (Cramer's Rule).
4. **Static Non-Snake Casing:** Methods leverage `#[allow(non_snake_case)]` to match textbook terminology.

## See also

<https://thekalmanfilter.com/kalman-filter-explained-simply/>

| S | description                     | type                | usage        |
| - | ------------------------------- | ------------------- | ------------ |
| x | state variable                  | n x 1 column vector | Output       |
| P | state covariance matrix         | n x n matrix        | Output       |
| z | measurement                     | m x 1 column vector | Input        |
| A | state transition matrix         | n x n matrix        | System Model |
| H | state-to-measurement matrix     | m x n matrix        | System Model |
| R | measurement covariance matrix   | m x m matrix        | Input        |
| Q | process noise covariance matrix | n x n matrix        | System Model |
| K | Kalman Gain                     | n x m               | Internal     |

For `AltitudeKalmanFilter`

n = 3 (velocity, altitude, and bias)
m = 1 (altitude)
