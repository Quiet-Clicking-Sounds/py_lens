use ndarray;
use ndarray::{Array2, Array3, ArrayView3};

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
/// returns: [f32; 2]
///
pub fn space_def(x: f32, y: f32, cx: f32, cy: f32, u: f32) -> [f32; 2] {
    let (xa, ya) = (x - cx, y - cy);
    let ang: (f32, f32) = ya.atan2(xa).sin_cos();
    let hyp = xa.hypot(ya);
    let hyp = if hyp.is_sign_negative() {
        ((hyp).sqrt()).sin().powi(2) * u
    } else {
        -((hyp.abs()).sqrt()).sin().powi(2) * u
    };
    return [hyp * ang.1 + x, hyp * ang.0 + y];
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
fn z_max_usize(n: f32, mx: usize) -> usize {
    mx.min(n.max(0f32) as usize)
}


/// see [z_max_usize()] for details
///
/// # Arguments
///
/// * `spd`: spacial position in format [f32,f32]
/// * `mx`: maximum size in x
/// * `my`: maximum size in y
///
/// returns: [usize; 2]
///
/// # Examples
///
/// ```
/// let pos: [usize; 2] = space_def_to_pos([13.3, 17.4], 15, 15)
/// assert_eq!([13,15], pos)
/// ```
fn space_def_to_pos(spd: [f32; 2], mx: usize, my: usize) -> [usize; 2] {
    [z_max_usize(spd[0], mx), z_max_usize(spd[1], my)]
}


///
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
pub fn apply_lens_rgb<'a>(image: &'a ArrayView3<'a, u8>, cx: f32, cy: f32, u: f32) -> Array3<u8> {
    let (mx, my) = (image.shape()[0]-1, image.shape()[1]-1);
    let im_shape: (usize, usize, usize) = (image.shape()[0], image.shape()[1], image.shape()[2]);

    let cx = if cx <= 1f32 { im_shape.0 as f32 * cx }  else { cx };
    let cy = if cy <= 1f32 { im_shape.1 as f32 * cy }  else { cy };

    let mut indices: Array2<[usize; 2]> = Array2::from_shape_fn((im_shape.0, im_shape.1),
                                                                |(a, b)| { [a, b] });

    indices.par_map_inplace(|xy| {
        *xy = space_def_to_pos(space_def(xy[0] as f32, xy[1] as f32, cx, cy, u), mx, my)
    });

    let mut out_img: Array3<u8> = Array3::zeros(im_shape);
    out_img.indexed_iter_mut().for_each(|((x, y, z), px)| {
        let xy = indices[[x, y]];
        *px = image[[xy[0], xy[1], z]]
    });


    out_img
}
