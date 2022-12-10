use ndarray::{ArrayView, Ix3};

pub type WinFunc<T> = fn(ArrayView<T, Ix3>) -> T;

pub trait NumConv {
    fn from_f64(f: f64) -> Self;
    fn as_u64(&self) -> u64;
    fn as_f64(&self) -> f64;
    fn clamp_rms_max(f: f64) -> Self;
}

impl NumConv for u8 {
    fn from_f64(f: f64) -> Self {
        f as u8
    }
    fn as_u64(&self) -> u64 {
        *self as u64
    }
    fn as_f64(&self) -> f64 {
        *self as f64
    }

    fn clamp_rms_max(f: f64) -> Self {
        Self::from_f64(f * 2.0)
    }
}

impl NumConv for u16 {
    fn from_f64(f: f64) -> Self {
        f as u16
    }
    fn as_u64(&self) -> u64 {
        *self as u64
    }
    fn as_f64(&self) -> f64 {
        *self as f64
    }


    fn clamp_rms_max(f: f64) -> Self {
        Self::from_f64(f * 2.0)
    }
}

impl NumConv for u32 {
    fn from_f64(f: f64) -> Self {
        f as u32
    }
    fn as_u64(&self) -> u64 {
        *self as u64
    }
    fn as_f64(&self) -> f64 {
        *self as f64
    }

    fn clamp_rms_max(f: f64) -> Self {
        Self::from_f64(f * 2.0)
    }
}

impl NumConv for u64 {
    fn from_f64(f: f64) -> Self {
        f as u64
    }
    fn as_u64(&self) -> u64 {
        *self
    }
    fn as_f64(&self) -> f64 {
        *self as f64
    }

    fn clamp_rms_max(f: f64) -> Self {
        Self::from_f64(f * 2.0)
    }
}

/// builtin rms with ndarray
pub fn stdev_ddof_0<T>(w: ArrayView<T, Ix3>) -> T
    where
        T: NumConv,
        T: Clone,
{
    let w = w.mapv(|elem: T| elem.as_f64());
    let w = w.std(0f64);
    T::from_f64(w)
}

/// builtin rms with ndarray
pub fn stdev_ddof_1<T>(w: ArrayView<T, Ix3>) -> T
    where
        T: NumConv,
        T: Clone,
{
    let w = w.mapv(|elem: T| elem.as_f64());
    let w = w.std(1f64);
    T::from_f64(w)
}

/// slightly faster than the ndarray rms, need to test more, but I like this one more...
pub fn faster_rms_u64_adding<T: NumConv>(w: ArrayView<T, Ix3>) -> T {
    let len_inv = (w.len() as f64).recip();
    let mean: f64 = w
        .iter()
        .fold(0u64, |a: u64, x: &T| a.saturating_add(x.as_u64())) as f64
        * len_inv;
    let flt: f64 = w
        .iter()
        .fold(0f64, |a: f64, x: &T| a + (x.as_f64() - mean).abs().powi(2));

    T::clamp_rms_max((flt * len_inv).sqrt())
}

#[cfg(test)]
mod tests {
    use crate::window;
    use ndarray::{Array3, Array4};
    use window::window_methods::*;

    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};


    fn not_a_hash((a, b, c): (usize, usize, usize)) -> u8 {
        ((476579u64 % (a * b * c + 1) as u64) % 256) as u8
    }

    fn generate_array4() -> Array4<u8> {
        Array4::from_shape_fn((1000, 8, 8, 1), |(a, b, c, d)| not_a_hash((b, c, d)))
    }

    fn generate_array3() -> Array3<u8> {
        Array3::from_shape_fn((8, 8, 1), not_a_hash)
    }

    #[test]
    fn test_stdev_ddof_1() {
        let a = generate_array3();
        let b = stdev_ddof_1(a.view());

        let mut hasher = DefaultHasher::new();
        b.hash(&mut hasher);
        assert_eq!(7541581120933061747, hasher.finish());
    }

    #[test]
    fn test_stdev_ddof_0() {
        let a = generate_array3();
        let b = stdev_ddof_0(a.view());

        let mut hasher = DefaultHasher::new();
        b.hash(&mut hasher);
        assert_eq!(7541581120933061747, hasher.finish());
    }

    #[test]
    fn test_faster_rms_u64_adding() {
        let a = generate_array3();
        let b = faster_rms_u64_adding(a.view());

        let mut hasher = DefaultHasher::new();
        b.hash(&mut hasher);
        assert_eq!(7541581120933061747, hasher.finish());
    }
}
