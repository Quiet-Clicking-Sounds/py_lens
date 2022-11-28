use std::fmt::{Display, Formatter};
use std::thread;
use ndarray::{
    ArrayView,
    Array3,
    AssignElem,
    Axis,
    Dim,
    Ix,
    Ix3,
    s,
    Shape,
};

use std::sync::mpsc;

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
    pub fn array_size(self, ar: &Array3<u8>) -> (Shape<Dim<[Ix; 3]>>, Dim<[Ix; 3]>) {
        let w = match self {
            WindowShape::Single(a) => { (a, a, 1) }
            WindowShape::Double(a, b) => { (a, b, 1) }
            WindowShape::Triple(a, b, c) => { (a, b, c) }
        };
        assert!(ar.shape()[0] >= w.0, "First Dimension of Array must be larger than of Window");
        assert!(ar.shape()[1] >= w.1, "Second Dimension of Array must be larger than of Window");
        assert!(ar.shape()[2] >= w.2, "Third Dimension of Array must be larger than of Window");
        let sh = ar.raw_dim();
        let dim: Dim<[Ix; 3]> = Dim([
            // note brackets matter, this is a - (b - 1)  NOT (a - b) - 1 this would be bad
            sh[0].saturating_sub(w.0.saturating_sub(1)),
            sh[1].saturating_sub(w.1.saturating_sub(1)),
            sh[2].saturating_sub(w.2.saturating_sub(1))
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
    pub fn create_v_splits(self, arr: &Array3<u8>) -> Vec<(usize, usize)> {
        let shape_0 = arr.shape()[0];
        let win_0 = match self {
            WindowShape::Single(a) => a,
            WindowShape::Double(a, _) => a,
            WindowShape::Triple(a, _, _) => a,
        };
        assert!(CORES < shape_0 - win_0, "Not enough splits for multithreading");
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
                if c_ == CORES { shape_0 } else { b.round() as usize + win_0.saturating_sub(1) }
            )
        };
        let v_splits_for_array: Vec<_> = (0..CORES).map(split_shape).collect();
        v_splits_for_array
    }
}

impl Display for WindowShape {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            WindowShape::Single(a) => { write!(f, "({})", a) }
            WindowShape::Double(a, b) => { write!(f, "({}, {})", a, b) }
            WindowShape::Triple(a, b, c) => { write!(f, "({}, {}, {})", a, b, c) }
        }
    }
}

#[allow(dead_code)]
const CORES: usize = 12;
#[allow(dead_code)]
const F32U8MAX: f32 = u8::MAX as f32;

#[allow(dead_code)]
type WinFunc = fn(ArrayView<u8, Ix3>) -> u8;


pub mod window_apply_methods {
    use ndarray::{ArrayView, Ix3};

    pub fn u8sum(w: ArrayView<u8, Ix3>) -> u8 {
        let mut u = 0u8;
        for i in w.iter() {
            u = u.wrapping_add(*i)
        }
        u
    }

    /// builtin stdev with ndarray
    pub fn stdev(w: ArrayView<u8, Ix3>) -> u8 {
        let w = w.mapv(|elem| elem as f32);
        let w = w.std(1f32);
        w as u8
    }

    /// slightly faster than the ndarray stdev, need to test more, but I like this one more...
    pub fn faster_stdev(w: ArrayView<u8, Ix3>) -> u8 {
        let w = w.mapv(|elem| elem as f32);
        let len_inv = (w.len() as f32).recip();
        let mean: f32 = w.iter().sum::<f32>() * len_inv;

        let flt: f32 = w.iter()
            .fold(0f32, |a, x| {
                a + (*x as f32 - mean).abs().powi(2)
            });

        (flt * len_inv).sqrt() as u8
    }

    pub fn average(w: ArrayView<u8, Ix3>) -> u8 {
        (w.sum() as f32 / w.len() as f32) as u8
    }
}


/// apply function where `fn(Array3<u8>)->u8` for moving stdev calculations or similar
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
/// ar2 = apply_over_window(arr, WindowShape::Single(3), window::stdev)
/// ar2.shape()
/// >>> [323,323,3]
///
/// ```
fn apply_over_window(arr: Array3<u8>, win_size: WindowShape, func: WinFunc) -> Array3<u8> {
    let (sh2, d) = win_size.array_size(&arr);
    // create windowed parts of the array
    let win = arr.windows(d);
    // create an uninitiated base array for the output, shape descried by windowed_array_size
    let mut un_arr = Array3::<u8>::zeros(sh2);

    // iter through the output array and the windowed array
    for (a, w) in un_arr.iter_mut().zip(win.into_iter()) {
        a.assign_elem(func(w)); // assignments for some reason, I think = was being unhelpful
    };
    un_arr
}


/// run n threads to compute the given function over a moving window of the array
///
/// Thread count: 12 on 12 thread cpu using stdev calulation over |  Total Pixels: 16777216
///  Multi Thread stdev calc over window:6 | Timed: 2.468437s | Shape: in: (4096, 4096, 3), out: [4091, 4091, 3]
///  Single Thread stdev calc over window:6 | Timed: 15.678387s | Shape: in: (4096, 4096, 3), out: [4091, 4091, 3]
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
pub fn thread_apply_over_window(input_array: Array3<u8>, win_size: WindowShape, func: WinFunc) -> Array3<u8> {

    // see WindowShape
    let v_splits_for_array = win_size.create_v_splits(&input_array);

    // thread return items keeper. Reminder: order here is important,
    // I'm not sending ordering information
    let mut thread_workers: Vec<_> = vec![];
    for (va, vb) in v_splits_for_array {
        let (tx, rx) = mpsc::channel();
        //let send_copy_of_window_size = window_size; // I dont think this is required

        // create a slice view of the array before sending it
        let pre_compute_sliced_array = input_array
            .slice(s![va..vb,..,..]);
        // needs ownership, probably possible to refactor that out
        let pre_compute_slice = pre_compute_sliced_array.to_owned();
        thread::spawn(move || {
            // thread open, do compute and send to `rx`
            let computed_array_output = apply_over_window(
                pre_compute_slice,
                win_size,
                func);
            tx.send(computed_array_output).unwrap();
        });
        // attach the new thread receiver to the worker vec
        thread_workers.push(rx);
    }
    // export all the threads once they're finished, must wait for all finished
    // otherwise we end up with things out of order
    let array_stacks: Vec<_> = thread_workers.iter()
        .map(|rx| rx.recv().unwrap()).collect();
    // views, concat doesn't like actual arrays
    let array_stacks_view: Vec<_> = array_stacks.iter().map(|a| a.view()).collect();
    // stack the arrays back into a single array, then return it
    let re_stacked_array = ndarray::concatenate(
        Axis(0),
        array_stacks_view.as_slice())
        .unwrap();

    re_stacked_array
}

#[cfg(test)]
mod tests {
    use std::ops::Sub;
    use ndarray::{Array3};
    use crate::window;
    use std::time;
    use window::window_apply_methods::*;
    use crate::window::WindowShape;

    static US_U8_MAX: usize = u8::MAX as usize;


    fn odd_func(a: usize, b: usize, c: usize) -> u8 {
        let a = a.rem_euclid(US_U8_MAX) as u8;
        let b = b.rem_euclid(US_U8_MAX) as u8;
        let c = c as u8;
        a ^ b ^ c
    }

    fn generate_array3(x: usize, y: usize) -> Array3<u8> {
        Array3::from_shape_fn((x, y, 3), |(a, b, c)| odd_func(a, b, c))
    }


    #[test]
    fn test_window_xor() {
        let sz: usize = 1080;
        let a1 = generate_array3(sz, sz);
        let a1sh = (a1.shape()[0], a1.shape()[1], a1.shape()[2]);
        let a2 = generate_array3(sz, sz);
        let a2sh = (a2.shape()[0], a2.shape()[1], a2.shape()[2]);

        let window = WindowShape::Single(6);

        println!("Total Pixels: {:?}", sz * sz);
        let t1 = time::Instant::now();
        let b1 = window::thread_apply_over_window(
            a1, window,
            faster_stdev,
        );
        println!("Multi Thread stdev calc over window:{} | Timed: {:?}s | Shape: in: {:?}, out: {:?}",
                 window,
                 time::Instant::now().sub(t1).as_secs_f32(),
                 a1sh, b1.shape());

        let t1 = time::Instant::now();
        let b2 = window::apply_over_window(
            a2, window,
            faster_stdev,
        );
        println!("Single Thread stdev calc over window:{} | Timed: {:?}s | Shape: in: {:?}, out: {:?}",
                 window,
                 time::Instant::now().sub(t1).as_secs_f32(),
                 a2sh, b2.shape());


        assert!(b1.eq(&b2))
    }

    #[test]
    fn bench_stdev() {
        fn shitty_bench(name: String, f: window::WinFunc) {
            let arr = Array3::from_shape_fn(
                (8, 8, 1),
                |(a, b, c)| (a + b + c) as u8,
            );
            let t1 = time::Instant::now();
            for _ in 0..10000 {
                let _ = f(arr.view());
            }
            println!("Timed: {:^12} - {:>12?}ns", name, time::Instant::now().sub(t1).as_nanos());
        }
        shitty_bench("stdev".into(), stdev);
        shitty_bench("mystd".into(), faster_stdev);
        shitty_bench("u8sum".into(), u8sum);
    }
}
