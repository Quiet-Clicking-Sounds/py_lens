use ndarray::Array3;
use ndarray::s;

use ndarray::AssignElem;
use ndarray::Axis;
use ndarray::Dim;
use ndarray::Ix;

use ndarray::Shape;
use num_traits::identities::Zero;
use std::fmt::{Display, Formatter};
use std::sync::mpsc;
use std::thread;

pub mod window_methods;

use crate::window::window_methods::NumConv;
use window_methods::WinFunc;

///
/// # Single(a)
/// used for a moving window with dimensions `[a, a, 1]` useful for images
///
/// # Double(a, b)
/// like Single, `[a, b, 1]` allows for rectangles
///
/// # Triple(a, b, c)
/// complete control over all sizes of the window, `[a, b, c]`
///
///
/// access to automatic sizing within using [`WindowShape::array_size`]
///
/// access to positional splits for threading using [`WindowShape::create_v_splits`]
///
#[derive(Copy, Clone)]
pub enum WindowShape {
    Single(usize),
    Double(usize, usize),
    Triple(usize, usize, usize),
}

impl WindowShape {
    ///
    ///
    /// # Arguments
    ///
    /// * `ar`: input array, used to get the input shape
    ///
    /// returns: (Shape<Dim<[usize; 3]>>, Dim<[usize; 3]>)
    ///
    /// # PANIC
    /// if any window size is greater than it's corresponding array size
    ///
    /// # Usage
    ///
    /// see: [`apply_over_window`]
    ///
    pub fn array_size<U>(self, ar: &Array3<U>) -> (Shape<Dim<[Ix; 3]>>, Dim<[Ix; 3]>) {
        let w = match self {
            WindowShape::Single(a) => (a, a, 1),
            WindowShape::Double(a, b) => (a, b, 1),
            WindowShape::Triple(a, b, c) => (a, b, c),
        };
        assert!(
            ar.shape()[0] >= w.0,
            "First Dimension of Array must be larger than of Window"
        );
        assert!(
            ar.shape()[1] >= w.1,
            "Second Dimension of Array must be larger than of Window"
        );
        assert!(
            ar.shape()[2] >= w.2,
            "Third Dimension of Array must be larger than of Window"
        );
        let sh = ar.raw_dim();
        let dim: Dim<[Ix; 3]> = Dim([
            // note brackets matter, this is a - (b - 1)  NOT (a - b) - 1 this would be bad
            sh[0].saturating_sub(w.0.saturating_sub(1)),
            sh[1].saturating_sub(w.1.saturating_sub(1)),
            sh[2].saturating_sub(w.2.saturating_sub(1)),
        ]);

        (Shape::from(dim), Dim([w.0, w.1, w.2]))
    }

    /// do not mix WindowShape instances
    ///
    /// # Arguments
    ///
    /// * `arr`: input array that will be used for window functions
    ///
    /// returns: Vec<(usize, usize), Global>
    ///
    /// # Errors:
    /// will panic if:
    /// ``` rust
    /// CORES > arr.shape()[0] - WindowShape[0]]
    /// ```
    ///
    /// # Examples
    ///
    /// See code for: [`thread_apply_over_window`]
    ///
    /// ``` rust
    /// let v_splits_for_array = win_size.create_v_splits(&input_array);
    /// ```
    pub fn create_v_splits<U>(self, arr: &Array3<U>) -> Vec<(usize, usize)> {
        let shape_0 = arr.shape()[0];
        let win_0 = match self {
            WindowShape::Single(a) => a,
            WindowShape::Double(a, _) => a,
            WindowShape::Triple(a, _, _) => a,
        };
        assert!(
            CORES < shape_0 - win_0,
            "Not enough splits for multithreading"
        );
        // TODO: Setup method of split even when CORES >  shape_0 - win_0.
        let v_split_size = (shape_0 - win_0.saturating_sub(1)) as f32 / CORES as f32;
        let split_shape = |c_: usize| -> (usize, usize) {
            let c = c_ as f32;
            let a = c * v_split_size;
            let b = (c + 1f32) * v_split_size;
            // last split needs to grab any leftover stuff, this just makes sure rounding errors
            // don't break things again, could be refactored out at some point

            (
                a.round() as usize,
                if c_ == CORES {
                    shape_0
                } else {
                    b.round() as usize + win_0.saturating_sub(1)
                },
            )
        };
        let v_splits_for_array: Vec<_> = (0..CORES).map(split_shape).collect();
        v_splits_for_array
    }
    #[allow(dead_code)]
    fn window_size(self) -> usize {
        match self {
            WindowShape::Single(a) => a * a,
            WindowShape::Double(a, b) => a * b,
            WindowShape::Triple(a, b, c) => b * a * c,
        }
    }
}

impl Display for WindowShape {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            WindowShape::Single(a) => {
                write!(f, "({})", a)
            }
            WindowShape::Double(a, b) => {
                write!(f, "({}, {})", a, b)
            }
            WindowShape::Triple(a, b, c) => {
                write!(f, "({}, {}, {})", a, b, c)
            }
        }
    }
}

#[allow(dead_code)]
const CORES: usize = 12;

/// apply function where `fn(Array3<u8>)->u8` for moving rms calculations or similar
/// return array will be smaller by the size of `s`
///
/// # Arguments
///
/// * `arr`: 3 dimensional array, image format (width, height, colour_value)
/// * `s`: size of moving window `[s,s,1]` (ignores z-depth of array for image editing purposes)
/// * `func`: `fn(Array3<u8>)->u8`
///
/// returns: ArrayBase<OwnedRepr<u8>, Dim<[usize; 3]>>
///
/// # Examples
///
/// ```
/// arr = Array3::from_image(image)
/// arr.shape()
/// >>> [325,325,3]
/// ar2 = apply_over_window(arr, WindowShape::Single(3), window::rms)
/// ar2.shape()
/// >>> [323,323,3]
///
/// ```
fn apply_over_window<T>(arr: Array3<T>, win_size: WindowShape, func: WinFunc<T>) -> Array3<T>
where
    T: Zero,
    T: NumConv,
    T: Clone,
{
    let (sh2, d) = win_size.array_size(&arr);
    // create windowed parts of the array
    let win = arr.windows(d);
    // create an uninitiated base array for the output, shape descried by windowed_array_size
    let mut un_arr = Array3::<T>::zeros(sh2);

    // iter through the output array and the windowed array
    for (a, w) in un_arr.iter_mut().zip(win.into_iter()) {
        a.assign_elem(func(w)); // assignments for some reason, I think = was being unhelpful
    }
    un_arr
}

/// run n threads to compute the given function over a moving window of the array
///
/// Thread count: 12 on 12 thread cpu using rms calculation over |  Total Pixels: 16777216
///  Multi Thread rms calc over window:6 | Timed: 2.468437s | Shape: in: (4096, 4096, 3), out: [4091, 4091, 3]
///  Single Thread rms calc over window:6 | Timed: 15.678387s | Shape: in: (4096, 4096, 3), out: [4091, 4091, 3]
///
/// # Arguments
///
/// * `input_array`:  3 dimensional array, image format (width, height, colour_value)
/// * `window_size`:  [`WindowShape`]
/// * `func`:  fn(Array3<u8>)->u8
///
/// returns: ArrayBase<OwnedRepr<u8>, Dim<[usize; 3]>>
///
/// # Examples
///
/// ```
///
/// ```
pub fn thread_apply_over_window<T>(
    input_array: Array3<T>,
    win_size: WindowShape,
    func: WinFunc<T>,
) -> Array3<T>
where
    T: Zero + NumConv + Clone + Copy + Send + 'static,
{
    // see WindowShape
    let v_splits_for_array = win_size.create_v_splits(&input_array);

    // thread return items keeper. Reminder: order here is important,
    // I'm not sending ordering information
    let mut thread_workers: Vec<_> = vec![];
    for (va, vb) in v_splits_for_array {
        let (tx, rx) = mpsc::channel();
        //let send_copy_of_window_size = window_size; // I dont think this is required

        // create a slice view of the array before sending it
        let pre_compute_sliced_array = input_array.slice(s![va..vb, .., ..]);
        // needs ownership, probably possible to refactor that out
        let pre_compute_slice = pre_compute_sliced_array.to_owned();
        thread::spawn(move || {
            // thread open, do compute and send to `rx`
            let computed_array_output = apply_over_window(pre_compute_slice, win_size, func);
            tx.send(computed_array_output).unwrap();
        });
        // attach the new thread receiver to the worker vec
        thread_workers.push(rx);
    }
    // export all the threads once they're finished, must wait for all finished
    // otherwise we end up with things out of order
    let array_stacks: Vec<_> = thread_workers.iter().map(|rx| rx.recv().unwrap()).collect();
    // views, concat doesn't like actual arrays
    let array_stacks_view: Vec<_> = array_stacks.iter().map(|a| a.view()).collect();
    // stack the arrays back into a single array, then return it
    let re_stacked_array = ndarray::concatenate(Axis(0), array_stacks_view.as_slice()).unwrap();

    re_stacked_array
}

#[cfg(test)]
mod tests {

    use crate::window;
    use ndarray::Array3;

    use crate::window::{thread_apply_over_window, WindowShape};
    use window::window_methods::*;

    fn generate_tst_array3u8() -> Array3<u8> {
        Array3::from_shape_fn((500, 500, 3), |(a, b, c): (usize, usize, usize)| {
            (a ^ b ^ c) as u8
        })
    }

    fn generate_tst_array3u16() -> Array3<u16> {
        Array3::from_shape_fn((500, 500, 3), |(a, b, c): (usize, usize, usize)| {
            (a ^ b ^ c) as u16
        })
    }

    fn generate_tst_array3u32() -> Array3<u32> {
        Array3::from_shape_fn((500, 500, 3), |(a, b, c): (usize, usize, usize)| {
            (a ^ b ^ c) as u32
        })
    }

    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    #[test]
    fn u8_test_with_std_array_rms_u64() {
        let test_array = generate_tst_array3u8();
        let win_shape = WindowShape::Triple(5, 5, 1);
        let out = thread_apply_over_window(test_array, win_shape, faster_rms_u64_adding);

        let mut hasher = DefaultHasher::new();
        out.hash(&mut hasher);
        assert_eq!(13253406290038557312, hasher.finish());
    }

    #[test]
    fn u8_test_with_std_array_stdev_ddof_1() {
        let test_array = generate_tst_array3u8();
        let win_shape = WindowShape::Triple(5, 5, 1);
        let out = thread_apply_over_window(test_array, win_shape, stdev_ddof_1);

        let mut hasher = DefaultHasher::new();
        out.hash(&mut hasher);
        assert_eq!(1348078754090702269, hasher.finish());
    }

    #[test]
    fn u8_test_with_std_array_stdev_ddof_0() {
        let test_array = generate_tst_array3u8();
        let win_shape = WindowShape::Triple(5, 5, 1);
        let out = thread_apply_over_window(test_array, win_shape, stdev_ddof_0);

        let mut hasher = DefaultHasher::new();
        out.hash(&mut hasher);
        assert_eq!(16665323007214972568, hasher.finish());
    }

    #[test]
    fn u16_test_with_std_array_rms_u64() {
        let test_array = generate_tst_array3u16();
        let win_shape = WindowShape::Triple(5, 5, 1);
        let out = thread_apply_over_window(test_array, win_shape, faster_rms_u64_adding);

        let mut hasher = DefaultHasher::new();
        out.hash(&mut hasher);
        assert_eq!(8798330982248845663, hasher.finish());
    }

    #[test]
    fn u16_test_with_std_array_stdev_ddof_1() {
        let test_array = generate_tst_array3u16();
        let win_shape = WindowShape::Triple(5, 5, 1);
        let out = thread_apply_over_window(test_array, win_shape, stdev_ddof_1);

        let mut hasher = DefaultHasher::new();
        out.hash(&mut hasher);
        assert_eq!(11549407550617260967, hasher.finish());
    }

    #[test]
    fn u16_test_with_std_array_stdev_ddof_0() {
        let test_array = generate_tst_array3u16();
        let win_shape = WindowShape::Triple(5, 5, 1);
        let out = thread_apply_over_window(test_array, win_shape, stdev_ddof_0);

        let mut hasher = DefaultHasher::new();
        out.hash(&mut hasher);
        assert_eq!(8077758819790012028, hasher.finish());
    }

    #[test]
    fn u32_test_with_std_array_rms_u64() {
        let test_array = generate_tst_array3u32();
        let win_shape = WindowShape::Triple(5, 5, 1);
        let out = thread_apply_over_window(test_array, win_shape, faster_rms_u64_adding);

        let mut hasher = DefaultHasher::new();
        out.hash(&mut hasher);
        assert_eq!(12289755379548528329, hasher.finish());
    }

    #[test]
    fn u32_test_with_std_array_stdev_ddof_1() {
        let test_array = generate_tst_array3u32();
        let win_shape = WindowShape::Triple(5, 5, 1);
        let out = thread_apply_over_window(test_array, win_shape, stdev_ddof_1);

        let mut hasher = DefaultHasher::new();
        out.hash(&mut hasher);
        assert_eq!(7961783534460445393, hasher.finish());
    }

    #[test]
    fn u32_test_with_std_array_stdev_ddof_0() {
        let test_array = generate_tst_array3u32();
        let win_shape = WindowShape::Triple(5, 5, 1);
        let out = thread_apply_over_window(test_array, win_shape, stdev_ddof_0);

        let mut hasher = DefaultHasher::new();
        out.hash(&mut hasher);
        assert_eq!(5942265300642722970, hasher.finish());
    }
}
