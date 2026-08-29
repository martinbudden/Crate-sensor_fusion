use core::ops::Mul;

use num_traits::Zero;

use vqm::{Matrix3x3Math, Matrix9, Matrix9x9, Matrix9x9Math, Vector3};

/// Kalman state vector of `f32` values<br>
pub type KalmanStateVector9f32 = KalmanStateVector9<f32>;
/// Kalman tate vector of `f64` values<br><br>
pub type KalmanStateVector9f64 = KalmanStateVector9<f64>;

/// Flattened representation of a 9-element state vector for Kalman filter matrix math.<br><br>
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C, align(64))]
#[allow(missing_docs)]
pub struct KalmanStateVector9<T> {
    pub pos: Vector3<T>,
    pub vel: Vector3<T>,
    pub bias: Vector3<T>,
}

impl<T> KalmanStateVector9<T>
where
    T: Copy,
{
    /// Constructor.
    #[inline]
    pub const fn new(v: (Vector3<T>, Vector3<T>, Vector3<T>)) -> Self {
        Self { pos: v.0, vel: v.1, bias: v.2 }
    }
}

impl<T> From<(Vector3<T>, Vector3<T>, Vector3<T>)> for KalmanStateVector9<T> {
    #[inline]
    fn from(v: (Vector3<T>, Vector3<T>, Vector3<T>)) -> Self {
        Self { pos: v.0, vel: v.1, bias: v.2 }
    }
}

/// Implement vector-by-scalar multiplication to scale the Kalman Gain.
impl Mul<f32> for KalmanStateVector9<f32> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: f32) -> Self::Output {
        Self { pos: self.pos * rhs, vel: self.vel * rhs, bias: self.bias * rhs }
    }
}

impl Mul<f64> for KalmanStateVector9<f64> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: f64) -> Self::Output {
        Self { pos: self.pos * rhs, vel: self.vel * rhs, bias: self.bias * rhs }
    }
}

impl<T> KalmanStateVector9<T>
where
    T: Copy + Zero + Matrix9x9Math + Matrix3x3Math + Mul<T, Output = T> + PartialEq,
{
    /// Calculates the outer product of two 9-element states in a compiler-friendly manner.
    #[inline]
    pub fn outer_product9(self, row: KalmanStateVector9<T>) -> Matrix9<T> {
        Matrix9::outer_product(self.pos, self.vel, self.bias, row.pos, row.vel, row.bias)
    }

    /// Calculates the outer product of two 9-element states in a compiler-friendly manner.
    #[inline]
    pub fn outer_product(self, row: KalmanStateVector9<T>) -> Matrix9x9<T> {
        Self::outer_product_associated(self, row)
    }

    /// Calculates the outer product of two 9-element states for COLUMN-MAJOR matrices (Cortex-M Edition).
    #[inline]
    fn outer_product_associated(col: KalmanStateVector9<T>, row: KalmanStateVector9<T>) -> Matrix9x9<T> {
        // Flatten row elements (scalar weights)
        let r = [row.pos.x, row.pos.y, row.pos.z, row.vel.x, row.vel.y, row.vel.z, row.bias.x, row.bias.y, row.bias.z];

        // Flatten column entries natively without any artificial 4-lane padding
        let c = [col.pos.x, col.pos.y, col.pos.z, col.vel.x, col.vel.y, col.vel.z, col.bias.x, col.bias.y, col.bias.z];

        let mut ret = <Matrix9x9<T>>::zero();

        // Process each column.
        // With the slice copies removed, the compiler can assign the entire `c` array
        // to CPU/FPU registers and stream them directly out to the matrix memory.
        for (c_idx, &scalar) in r.iter().enumerate().take(9) {
            let ret_col = &mut ret[c_idx * 9..(c_idx + 1) * 9];

            // Direct scalar assignment. LLVM unrolls this perfectly and generates
            // branchless, pipelined single-cycle hardware float multiplications.
            ret_col[0] = c[0] * scalar;
            ret_col[1] = c[1] * scalar;
            ret_col[2] = c[2] * scalar;
            ret_col[3] = c[3] * scalar;
            ret_col[4] = c[4] * scalar;
            ret_col[5] = c[5] * scalar;
            ret_col[6] = c[6] * scalar;
            ret_col[7] = c[7] * scalar;
            ret_col[8] = c[8] * scalar;
        }

        ret
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn _is_normal<T: Sized + Send + Sync + Unpin>() {}
    fn is_full<T: Sized + Send + Sync + Unpin + Copy + Clone + Default + PartialEq>() {}

    #[test]
    fn normal_types() {
        is_full::<KalmanStateVector9f32>();
    }
}
