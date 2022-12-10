use ndarray::{Array2, Array3, ArrayView3};

mod point_compute;

pub use point_compute::{ComputePoint, WaveLine, WavePoint, StarPattern};
///
///
/// # Arguments
///
/// * `x`: current x position
/// * `y`: current y position
/// * `cx`: centre unit position of the wave in x
/// * `cy`: centre unit position of the wave in y
/// * `u`: modifier for the strength of the wave
///
/// returns: [f64; 2]
///
pub fn point_start_wave(x: f64, y: f64, cx: f64, cy: f64, u: f64) -> [f64; 2] {
    let xa = x - cx;
    let ya = y - cy;
    let ang: (f64, f64) = ya.atan2(xa).sin_cos();
    let hyp = xa.hypot(ya);
    let hyp = if hyp.is_sign_negative() {
        ((hyp).sqrt()).sin().powi(2) * u
    } else {
        -((hyp.abs()).sqrt()).sin().powi(2) * u
    };
    [hyp * ang.1 + x, hyp * ang.0 + y]
}

///
///
/// # Arguments
///
/// * `n`: position coordinate in float32
/// * `mx`: maximum allowed coordinate in usize
///
/// returns: usize
///
/// # Examples
///
/// ```
/// let position = z_max_usize(13.4, 15);
/// assert_eq!(13, position);
///
/// let position = z_max_usize(17.3, 15);
/// assert_eq!(15, position);
///
/// let position = z_max_usize(-2.2, 15);
/// assert_eq!(0, position);
///
/// ```
fn z_max_usize(n: f64, mx: usize) -> usize {
    mx.min(n.max(0f64) as usize)
}

/// see [z_max_usize()] for details
///
/// # Arguments
///
/// * `spd`: spacial position in format [f64,f64]
/// * `mx`: maximum size in x
/// * `my`: maximum size in y
///
/// returns: (usize, usize)
///
/// # Examples
///
/// ```
/// let pos: (usize, usize) = space_def_to_pos([13.3, 17.4], 15, 15);
/// assert_eq!([13,15], pos);
/// ```
fn space_def_to_pos(spd: (f64, f64), mx: usize, my: usize) -> (usize, usize) {
    (z_max_usize(spd.0, mx), z_max_usize(spd.1, my))
}

///
///
/// # Arguments
///
/// * `image`: image array in format (width, height rgb)
/// * `wave_method`:
///
/// returns: ArrayBase<OwnedRepr<u8>, Dim<[usize; 3]>>
///
pub fn lens_rgb<'a, T>(image: &'a ArrayView3<'a, u8>, mut wave_method: T) -> Array3<u8>
where
    T: ComputePoint,
    T: Send,
    T: Sync,
{
    let (mx, my) = (image.shape()[0] - 1, image.shape()[1] - 1);
    wave_method.setup_for_new_image(mx, my);
    let im_shape: (usize, usize, usize) = (image.shape()[0], image.shape()[1], image.shape()[2]);

    let mut indices: Array2<(usize, usize)> =
        Array2::from_shape_fn((im_shape.0, im_shape.1), |(a, b)| (a, b));

    indices.par_map_inplace(|xy| {
        *xy = space_def_to_pos(wave_method.point_shift(xy.0 as f64, xy.1 as f64), mx, my)
    });

    let mut out_img: Array3<u8> = Array3::zeros(im_shape);
    out_img.indexed_iter_mut().for_each(|((x, y, z), px)| {
        let xy = indices[[x, y]];
        *px = image[[xy.0, xy.1, z]]
    });
    out_img
}

/// old, but it does work
///
/// # Arguments
///
/// * `image`: image array in format (width, height rgb)
/// * `cx`: positional centre of wave format in x
/// * `cy`: positional centre of wave format in y
/// * `u`: modifier for strength of the wave form
///
/// returns: ArrayBase<OwnedRepr<u8>, Dim<[usize; 3]>>
///
pub fn apply_lens_rgb<'a>(image: &'a ArrayView3<'a, u8>, cx: f64, cy: f64, u: f64) -> Array3<u8> {
    let (mx, my) = (image.shape()[0]-1, image.shape()[1]-1);
    let im_shape: (usize, usize, usize) = (image.shape()[0], image.shape()[1], image.shape()[2]);

    let cx = if cx <= 1f64 { im_shape.0 as f64 * cx }  else { cx };
    let cy = if cy <= 1f64 { im_shape.1 as f64 * cy }  else { cy };

    let mut indices: Array2<(usize,usize)> = Array2::from_shape_fn((im_shape.0, im_shape.1),
                                                                |(a, b)| { (a, b) });

    indices.par_map_inplace(|xy| {
        let psw = point_start_wave(xy.0 as f64, xy.1 as f64, cx, cy, u);
        *xy = space_def_to_pos((psw[0],psw[1]), mx, my);
    });

    let mut out_img: Array3<u8> = Array3::zeros(im_shape);
    out_img.indexed_iter_mut().for_each(|((x, y, z), px)| {
        let xy = indices[[x, y]];
        *px = image[[xy.0, xy.1, z]]
    });


    out_img
}



#[cfg(test)]
mod tests {
    use crate::lens;
    use crate::lens::point_compute;
    use ndarray::Array3;


    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    #[test]
    fn test_z_max_usize() {
        let pos = lens::z_max_usize(13.4f64, 15usize);
        assert_eq!(13usize, pos);
        let pos = lens::z_max_usize(17.3f64, 15usize);
        assert_eq!(15usize, pos);
        let pos = lens::z_max_usize(-2.2f64, 15usize);
        assert_eq!(0usize, pos);
    }

    #[test]
    fn test_space_def_to_pos() {
        let pos: (usize, usize) = lens::space_def_to_pos((13.3, 17.4), 15, 15);
        assert_eq!((13, 15), pos);
    }

    #[test]
    fn test_space_def() {
        let cx = 13f64;
        let cy = 13f64;
        let u = 5f64;

        let lens_tests = [
            ((3.4318097,3.4318097),(4.356793649078088,4.356793649078088)),
            ((13.0,14.9051285),(13.0,10.084455622437533)),
            ((219.99794,145.99867),(219.99586176403056,145.99733470835469)),
            ((1119.5486,1555.3707),(1119.0782803611603,1554.7151415992382)),
            ((3.4318097,3.4318097),(4.356793649078088,4.356793649078088)),
        ];
        for (a,b) in lens_tests{
            let arg = lens::point_start_wave(a.0, a.1, cx, cy, u);
            let b0 = (b.0 - arg[0]).abs() ;
            let b1 = (b.1 - arg[1]).abs() ;

            assert!(
                (b0 <= f64::EPSILON) & (b1<= f64::EPSILON),
                "In: {:?} {:?}\nOut: {:?} {:?}\n mismatch: {:?} {:?}",
                a,b , arg[0],arg[1],b0,b1
                );
        };
    }

    static US_U8_MAX: usize = u8::MAX as usize;

    fn odd_func(a: usize, b: usize, c: usize) -> u8 {
        let a = a.rem_euclid(US_U8_MAX) as u8;
        let b = b.rem_euclid(US_U8_MAX) as u8;
        let c = c as u8;
        a ^ b ^ c
    }

    #[test]
    fn test_trait_point_lens_rgb() {
        let cx = 13f64;
        let cy = 13f64;
        let u = 15f64;
        let arr: Array3<u8> = Array3::from_shape_fn((50, 50, 3), |(a, b, c)| odd_func(a, b, c));
        let trait_part = point_compute::WavePoint::new((cx, cy), u);
        let arr = lens::lens_rgb(&arr.view(), trait_part);

        let mut hasher = DefaultHasher::new();
        arr.hash(&mut hasher);
        assert_eq!(3681281280540927891, hasher.finish());
    }

    //#[test]
    fn test_trait_line_lens_rgb() {
        let cx = (0f64, 0f64);
        let _cy = (10f64, 7f64);
        let u = 15f64;
        let arr: Array3<u8> = Array3::from_shape_fn((50, 50, 3), |(a, b, c)| odd_func(a, b, c));
        let trait_part = point_compute::WaveLine::new(cx, 4.4, u);
        let arr = lens::lens_rgb(&arr.view(), trait_part);


        let mut hasher = DefaultHasher::new();
        arr.hash(&mut hasher);
        todo!();

        assert_eq!(0, hasher.finish());
    }
}
