#![allow(clippy::inline_always)]
#![allow(clippy::many_single_char_names)]

#[cfg(feature = "simd")]
use core::simd::{f32x4, simd_swizzle};

use vqm::{Quaternion, Vector3};

/// Math functions for `SensorFusion`, using **SIMD** accelerations for `f32`.
pub trait SensorFusionMath: Sized {
    fn estimate_gravity(q: Quaternion<Self>) -> Vector3<Self>;
    fn derivative(q: Quaternion<Self>, gyro: Vector3<Self>) -> Quaternion<Self>;
    fn madgwick_step_acc(q: Quaternion<Self>, acc: Vector3<Self>, max_acc_magnitude_squared: Self) -> Quaternion<Self>;
    fn madgwick_step_acc_mag(
        q: Quaternion<Self>,
        acc: Vector3<Self>,
        mag: Vector3<Self>,
        max_acc_magnitude_squared: Self,
    ) -> Quaternion<Self>;
}

impl SensorFusionMath for f32 {
    #[inline(always)]
    fn estimate_gravity(q: Quaternion<Self>) -> Vector3<Self> {
        Vector3 {
            x: 2.0 * (q.x * q.z - q.w * q.y),
            y: 2.0 * (q.y * q.z + q.w * q.x),
            z: q.w * q.w - q.x * q.x - q.y * q.y + q.z * q.z,
        }
    }

    #[inline(always)]
    fn derivative(q: Quaternion<Self>, gyro: Vector3<Self>) -> Quaternion<Self> {
        #[cfg(feature = "simd")]
        {
            /*// Load q: [w, x, y, z]
            let q_v: f32x4 = unsafe { core::mem::transmute(q) };

            // Load gyro: [x, y, z, padding] -> Swizzle to [0, x, y, z]
            let g_raw: f32x4 = unsafe { core::mem::transmute(gyro) };
            //let g_v = simd_swizzle!(g_raw, [f32x4::splat(0.0)], [4, 0, 1, 2]);

            // Efficiently shift: [x, y, z, 0] -> [0, x, y, z]
            // We rotate right and then zero out the 'w' (index 0)
            let g_v = simd_swizzle!(g_raw, [3, 0, 1, 2]) * f32x4::from_array([0.0, 1.0, 1.0, 1.0]);

            // Hamilton Product (q * g) for [w, x, y, z] layout:
            // w_out = -x1*gx - y1*gy - z1*gz
            // x_out =  w1*gx + y1*gz - z1*gy
            // y_out =  w1*gy - x1*gz + z1*gx
            // z_out =  w1*gz + x1*gy - y1*gx

            // Row A: [w, w, w, w] * [0, gx, gy, gz]
            let res = simd_swizzle!(q_v, [0, 0, 0, 0]) * g_v;

            // Row B: [x, -x, -x, -x] * [gx, 0, gz, gy]
            // (Using signs and swizzles to match the Hamilton rows)
            let x_part = simd_swizzle!(q_v, [1, 1, 1, 1])
                * simd_swizzle!(g_v, [1, 0, 3, 2])
                * f32x4::from_array([-1.0, 1.0, -1.0, 1.0]);

            // Row C: [y, y, -y, y] * [gy, gz, 0, gx]
            let y_part = simd_swizzle!(q_v, [2, 2, 2, 2])
                * simd_swizzle!(g_v, [2, 3, 0, 1])
                * f32x4::from_array([-1.0, 1.0, 1.0, -1.0]);

            // Row D: [z, z, z, -z] * [gz, gy, gx, 0]
            let z_part = simd_swizzle!(q_v, [3, 3, 3, 3])
                * simd_swizzle!(g_v, [3, 2, 1, 0])
                * f32x4::from_array([-1.0, -1.0, 1.0, 1.0]);

            let q_dot = (res + x_part + y_part + z_part) * f32x4::splat(0.5);

            unsafe { core::mem::transmute(q_dot) }*/
            let q_v = f32x4::from(q);
            let g_raw = f32x4::from(gyro);

            // Shift [x, y, z, pad] to [0, x, y, z] and zero the w lane
            let g_v = simd_swizzle!(g_raw, [3, 0, 1, 2]) * f32x4::from_array([0.0, 1.0, 1.0, 1.0]);

            // Parallel Hamilton Calculation
            let w1 = simd_swizzle!(q_v, [0, 0, 0, 0]);
            let x1 = simd_swizzle!(q_v, [1, 1, 1, 1]);
            let y1 = simd_swizzle!(q_v, [2, 2, 2, 2]);
            let z1 = simd_swizzle!(q_v, [3, 3, 3, 3]);

            let g_w = g_v; // [0, gx, gy, gz]
            let g_x = simd_swizzle!(g_v, [1, 0, 3, 2]); // [gx, 0, gz, gy]
            let g_y = simd_swizzle!(g_v, [2, 3, 0, 1]); // [gy, gz, 0, gx]
            let g_z = simd_swizzle!(g_v, [3, 2, 1, 0]); // [gz, gy, gx, 0]

            let res = (w1 * g_w)
                + (x1 * g_x * f32x4::from_array([-1.0, 1.0, -1.0, 1.0]))
                + (y1 * g_y * f32x4::from_array([-1.0, 1.0, 1.0, -1.0]))
                + (z1 * g_z * f32x4::from_array([-1.0, -1.0, 1.0, 1.0]));

            let q_dot = res * f32x4::splat(0.5);
            q_dot.into()
        }
        #[cfg(not(feature = "simd"))]
        {
            Quaternion {
                w: (-q.x * gyro.x - q.y * gyro.y - q.z * gyro.z) * 0.5,
                x: (q.w * gyro.x - q.z * gyro.y + q.y * gyro.z) * 0.5,
                y: (q.z * gyro.x + q.w * gyro.y - q.x * gyro.z) * 0.5,
                z: (-q.y * gyro.x + q.x * gyro.y + q.w * gyro.z) * 0.5,
            }
        }
    }

    /// Features of this implementation:
    ///
    /// 1. Parallel Throughput: Instead of 12 separate floating-point multiplications and 8 additions,
    ///    the SIMD unit performs 3 vector multiplications and 2 vector additions.
    ///
    /// 2. Instruction Density: The `simd_swizzle!` maps directly to the VREV or VMOV instructions on the M33.
    ///
    /// 3. Register Reuse: `q_v` stays in its SIMD register the entire time.
    ///    The compiler will likely use VFMA (Vector Fused Multiply-Add) to combine the terms,
    ///    meaning this whole function could resolve in under 15 clock cycles.
    ///
    #[inline(always)]
    fn madgwick_step_acc(q: Quaternion<Self>, acc: Vector3<Self>, max_acc_magnitude_squared: Self) -> Quaternion<Self> {
        use num_traits::Zero;
        use vqm::MathMethods;

        let acc_magnitude_squared = acc.norm_squared();
        // Acceleration is an unreliable indicator of orientation when in high-g maneuvers,
        // so exclude it from the calculation in these cases
        let mut a = acc;
        if acc_magnitude_squared > max_acc_magnitude_squared || acc_magnitude_squared == 0.0 {
            a = Vector3::zero();
        } else {
            a *= acc_magnitude_squared.sqrt_reciprocal();
        }
        #[cfg(feature = "simd")]
        {
            let q_v = f32x4::from(q);

            // Calculate wz_common
            let wz_common = 2.0 * (q.x * q.x + q.y * q.y);

            // Calculate xy_common (w*w + z*z - 1.0 + 2.0*wz_common + a.z)
            let xy_common = 2.0 * (q.w * q.w + q.z * q.z - 1.0 + 2.0 * wz_common + a.z);

            // Calculate common term: [w, x, y, z] * [qq, xy_common, xy_common, qq]
            let common_scalars = f32x4::from_array([wz_common, xy_common, xy_common, wz_common]);
            let common = q_v * common_scalars;

            // Calculate ax term: [y, -z, w, -x] * ax
            // Term 2: [y, -z, w, -x] * ax
            // Indices needed: y=2, z=3, w=0, x=1
            let ax_q_swiz = simd_swizzle!(q_v, [2, 3, 0, 1]);
            let ax_signs = f32x4::from_array([1.0, -1.0, 1.0, -1.0]);
            let ax = (ax_q_swiz * ax_signs) * f32x4::splat(a.x);

            // Calculate ay term: [-x, -w, -z, -y] * ay
            // Term 3: [-x, -w, -z, -y] * ay
            // Indices needed: x=1, w=0, z=3, y=2
            let ay_q_swiz = simd_swizzle!(q_v, [1, 0, 3, 2]);
            let ay_signs = f32x4::splat(-1.0);
            let ay = (ay_q_swiz * ay_signs) * f32x4::splat(a.y);

            // Combine: step = common + ax + ay
            let ret_v = common + ax + ay;

            ret_v.into()
        }
        #[cfg(not(feature = "simd"))]
        {
            let wz_common = 2.0 * (q.x * q.x + q.y * q.y);
            let xy_common = 2.0 * (q.w * q.w + q.z * q.z - 1.0 + 2.0 * wz_common + a.z);
            Quaternion {
                w: q.w * wz_common + q.y * a.x - q.x * a.y,
                x: q.x * xy_common - q.z * a.x - q.w * a.y,
                y: q.y * xy_common + q.w * a.x - q.z * a.y,
                z: q.z * wz_common - q.x * a.x - q.y * a.y,
            }
        }
    }

    #[inline(always)]
    fn madgwick_step_acc_mag(
        q: Quaternion<Self>,
        acc: Vector3<Self>,
        mag: Vector3<Self>,
        max_acc_magnitude_squared: Self,
    ) -> Quaternion<Self> {
        use num_traits::Zero;
        use vqm::MathMethods;

        let acc_magnitude_squared = acc.norm_squared();
        // Acceleration is an unreliable indicator of orientation when in high-g maneuvers,
        // so exclude it from the calculation in these cases
        let mut a = acc;
        if acc_magnitude_squared > max_acc_magnitude_squared || acc_magnitude_squared == 0.0 {
            a = Vector3::zero();
        } else {
            a *= acc_magnitude_squared.sqrt_reciprocal();
        }

        let m = mag.normalize();

        // make copies of the components of q to simplify the algebraic expressions
        let q0 = q.w;
        let q1 = q.x;
        let q2 = q.y;
        let q3 = q.z;
        // Auxiliary variables to avoid repeated arithmetic
        let q0q0 = q0 * q0;
        let q0q1 = q0 * q1;
        let q0q2 = q0 * q2;
        let q0q3 = q0 * q3;
        let q1q1 = q1 * q1;
        let q1q2 = q1 * q2;
        let q1q3 = q1 * q3;
        let q2q2 = q2 * q2;
        let q2q3 = q2 * q3;
        let q3q3 = q3 * q3;

        let q1q1_plus_q2q2 = q1q1 + q2q2;
        let q2q2_plus_q3q3 = q2q2 + q3q3;

        // Reference direction of Earth's magnetic field
        let h = Vector3 {
            x: m.x * (q0q0 + q1q1 - q2q2_plus_q3q3) + 2.0 * (m.y * (q1q2 - q0q3) + m.z * (q0q2 + q1q3)),
            y: (m.x * (q0q3 + q1q2) + m.y * (q0q0 - q1q1 + q2q2 - q3q3) + m.z * (q2q3 - q0q1)) * 2.0,
            z: 0.0,
        };

        let bx_bx = h.x * h.x + h.y * h.y;
        let b = Vector3 {
            x: bx_bx.sqrt(),
            y: 0.0,
            z: 2.0 * (m.x * (q1q3 - q0q2) + m.y * (q0q1 + q2q3)) + m.z * (q0q0 - q1q1_plus_q2q2 + q3q3),
        };

        let a_dash = Vector3 { x: a.x + m.x * b.z, y: a.y + m.y * b.z, z: 0.0 };
        let bz_bz = b.z * b.z;
        let _4bx_bz = 4.0 * b.x * b.z;

        let m_bx = m * b.x;
        let mz_bz = m.z * b.z;

        let sum_squares_minus_one = q0q0 + q1q1_plus_q2q2 + q3q3 - 1.0;
        let xy_common = sum_squares_minus_one + q1q1_plus_q2q2;
        let yz_common = sum_squares_minus_one + q2q2_plus_q3q3;
        let wz_common = q1q1_plus_q2q2 * (1.0 + bz_bz) + bx_bx;

        // Gradient decent algorithm corrective step
        #[allow(clippy::used_underscore_binding)]
        Quaternion {
            w: q0 * 2.0 * (wz_common * q2q2_plus_q3q3) - q1 * a_dash.y
                + q2 * (a_dash.x - m_bx.z)
                + q3 * (m_bx.y - _4bx_bz * q0q1),

            x: -q0 * a_dash.y + q1 * 2.0 * (xy_common * (1.0 + 2.0 * bz_bz) + mz_bz + bx_bx * q2q2_plus_q3q3 + a.z)
                - q2 * m_bx.y
                - q3 * (a_dash.x + m_bx.z + _4bx_bz * (0.5 * sum_squares_minus_one + q1q1)),

            y: q0 * (a_dash.x - m_bx.z) - q1 * m_bx.y
                + q2 * 2.0 * (xy_common * (1.0 + bz_bz) + mz_bz + m_bx.x + bx_bx * yz_common + a.z)
                - q3 * (a_dash.y + _4bx_bz * q1q2),

            z: q0 * m_bx.y - q1 * (a_dash.x + m_bx.z + _4bx_bz * (0.5 * sum_squares_minus_one + q3q3)) - q2 * a_dash.y
                + q3 * 2.0 * (wz_common + m_bx.x * yz_common),
        }
        .normalize()
    }
}

impl SensorFusionMath for f64 {
    #[inline(always)]
    fn estimate_gravity(q: Quaternion<Self>) -> Vector3<Self> {
        Vector3 {
            x: 2.0 * (q.x * q.z - q.w * q.y),
            y: 2.0 * (q.y * q.z + q.w * q.x),
            z: q.w * q.w - q.x * q.x - q.y * q.y + q.z * q.z,
        }
    }

    #[inline(always)]
    fn derivative(q: Quaternion<Self>, gyro_rps: Vector3<Self>) -> Quaternion<Self> {
        Quaternion {
            w: (-q.x * gyro_rps.x - q.y * gyro_rps.y - q.z * gyro_rps.z) * 0.5,
            x: (q.w * gyro_rps.x + q.y * gyro_rps.z - q.z * gyro_rps.y) * 0.5,
            y: (q.w * gyro_rps.y - q.x * gyro_rps.z + q.z * gyro_rps.x) * 0.5,
            z: (q.w * gyro_rps.z + q.x * gyro_rps.y - q.y * gyro_rps.x) * 0.5,
        }
    }

    #[inline(always)]
    fn madgwick_step_acc(q: Quaternion<Self>, acc: Vector3<Self>, max_acc_magnitude_squared: Self) -> Quaternion<Self> {
        use num_traits::Zero;
        use vqm::MathMethods;

        let acc_magnitude_squared = acc.norm_squared();
        // Acceleration is an unreliable indicator of orientation when in high-g maneuvers,
        // so exclude it from the calculation in these cases
        let mut a = acc;
        if acc_magnitude_squared > max_acc_magnitude_squared || acc_magnitude_squared == 0.0 {
            a = Vector3::zero();
        } else {
            a *= acc_magnitude_squared.sqrt_reciprocal();
        }
        let wz_common = 2.0 * (q.x * q.x + q.y * q.y);
        let xy_common = 2.0 * (q.w * q.w + q.z * q.z - 1.0 + 2.0 * wz_common + a.z);
        Quaternion {
            w: q.w * wz_common + q.y * a.x - q.x * a.y,
            x: q.x * xy_common - q.z * a.x - q.w * a.y,
            y: q.y * xy_common + q.w * a.x - q.z * a.y,
            z: q.z * wz_common - q.x * a.x - q.y * a.y,
        }
    }

    /// Not implemented for `f64`.
    #[inline(always)]
    fn madgwick_step_acc_mag(
        q: Quaternion<Self>,
        _acc: Vector3<Self>,
        _mag: Vector3<Self>,
        _max_acc_magnitude_squared: Self,
    ) -> Quaternion<Self> {
        q
    }
}
