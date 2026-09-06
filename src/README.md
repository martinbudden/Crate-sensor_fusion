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
│  2. Execute    ──► predict_covariance()  [P = F * E * Fᵀ + Q]  │
└────────────────────────────────────────────────────────────────┘
                            │
                            ▼
    Asynchronous Data Packets Available?
    ├── YES (Baro SPI @ ~40Hz)  ──► correct_barometer(z)  [S = P₂₂ + R]
    └── YES (GPS UART @ ~5Hz)   ──► correct_gps(x, y, z)  [S = H*P*Hᵀ + R]
```

## 3. Architectural Rule: The Accelerometer is NOT a Measurement Update

A common point of confusion is searching for an `update_accelerometer` function.
In a kinematic navigation filter, the accelerometer acts as the **Control Input (u)**
during the time-propagation step, pushing the physics model forward. Absolute reference sensors
(**GPS, Barometer**) act as **Measurements (z)** to strip away the integration drift.

## 4. Matrix Transformations

### Covariance Time Propagation: `P = F * E * Fᵀ + Q`

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

1. **Pass-by-Value API:** Data structures arrive via registers (`FPU s0-s31`), bypassing stack pointer overhead.
2. **Analytic Inversion:** Multi-dimensional GPS steps leverage a zero-loop 3x3 determinant inverse (Cramer's Rule).
3. **Non-Snake Case:** Methods use `#[allow(non_snake_case)]` to match textbook terminology.

## See also

<https://thekalmanfilter.com/kalman-filter-explained-simply/>

|   | description                                           | type                | usage        |
| - | ----------------------------------------------------- | ------------------- | ------------ |
| z | measurement                                           | m x 1 column vector | Input        |
| R | measurement noise covariance                          | m x m matrix        | Input        |
| x | state vector                                          | n x 1 column vector | Output       |
| P | state covariance matrix                               | n x n matrix        | System Model |
| F | state transition matrix                               | n x n matrix        | System Model |
| S | innovation covariance matrix                          | n x n matrix        | System Model |
| H | state-to-measurement matrix<br>aka observation matrix | m x n matrix        | System Model |
| Q | process noise covariance                              | n x n matrix        | System Model |
| K | Kalman Gain                                           | n x m               | Internal     |

For a 1D KalmanFilter n = 3, m = 1,
For a 2D KalmanFilter n = 6, m = 2,
For a 3D KalmanFilter n = 9, m = 3,

Note n = 3*m, since we have (velocity, altitude, bias) for each dimension.

|   |type          | 1D  | 2D  | 2D opt | 3D  | 3D opt |
| - | ------------ | --- | --- | -------| --- | ------ |
| x | n x 1 vector | 3x1 | 6x1 | 6x1    | 9x1 | 9x1    |
| R | m x m matrix | 1x1 | 2x2 | 2x1    | 3x3 | 3x1    |
| z | m x 1 vector | 1   | 2   | 2      | 3   | 3      |
| P | n x n matrix | 3x3 | 6x6 | sparse | 9x9 | sparse |
| F | n x n matrix | 3x3 | 6x6 | sparse | 9x9 | sparse |
| S | n x n matrix | 3x3 | 6x6 | 2x2    | 9x9 | 3x3    |
| H | m x n matrix | 1x3 | 2x6 | sparse | 3x9 | sparse |
| Q | n x n matrix | 3x3 | 6x6 | 2x1    | 9x9 | 2x1    |
| K | n x m matrix | 3x1 | 6x2 | 3xM2x2 | 9x2 | 3xM3x3 |

In the optimized form the following is used
For 3D:

`R` is `Vector3` rather than `Matrix3x3`.
`P` and `F` are `Matrix3x3xM3x3` rather than `Matrix9x9` and are treated as sparse matrices for calculation.
`S` is `Matrix3x3` rather than `Matrix9x9`, since `S` must be inverted this is a considerable optimization.
`H` is never explicitly created, but 3x3 submatrices of `H` are used in calculations.
`Q` is the two scalars `Q_velocity` and `Q_bias` rather than `Matrix9x9`.
`K` is implemented as three `Matrix3x3`s.

### The standard Kalman filter equations

```text
Predict: 
    x = F * x, 
    P_new = F * P * Fᵀ + Q

Correct: 
    S = H * P * Hᵀ + R
    K = P * Hᵀ * S^-1
    error = z - H * x, 
    x += K * error,
    P -= (K * H) * P
```
