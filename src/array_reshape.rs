use ndarray::Array2;
use ndarray::Array3;
use ndarray::ArrayView2;
use ndarray::ArrayView3;

const ORDER: u8 = 64;

pub fn image_to_line<'a>(image: &'a ArrayView3<'a, u8>) -> Array2<u8> {
    let im_shape = image.shape();
    let long_side = im_shape[0].max(im_shape[1]).next_power_of_two();
    let mut flat_img: Array2<u8> = Array2::zeros((long_side.pow(2), im_shape[2]));

    image.indexed_iter().for_each(|((x, y, z), &c)| {
        let t = fast_hilbert::xy2h(x as u32, y as u32, ORDER) as usize;
        flat_img[[t, z]] = c
    });

    flat_img
}

pub fn line_to_image<'a>(
    image: &'a ArrayView2<'a, u8>,
    imshape1: usize,
    imshape2: usize,
) -> Array3<u8> {
    let mut out_img: Array3<u8> = Array3::zeros((imshape1, imshape2, image.shape()[1]));

    out_img.indexed_iter_mut().for_each(|((x, y, z), c)| {
        let t = fast_hilbert::xy2h(x as u32, y as u32, ORDER) as usize;
        *c = image[[t, z]];
    });
    out_img
}

#[cfg(test)]
mod tests {
    use crate::array_reshape;
    use ndarray::Array3;

    static US_U8_MAX: usize = u8::MAX as usize;

    fn odd_func(a: usize, b: usize, c: usize) -> u8 {
        let a = a.rem_euclid(US_U8_MAX) as u8;
        let b = b.rem_euclid(US_U8_MAX) as u8;
        let c = c as u8;
        a ^ b ^ c
    }

    #[test]
    fn image_to_line_to_image_square() {
        let (sh1, sh2, sh3) = (50usize, 50usize, 3usize);
        let arr: Array3<u8> = Array3::from_shape_fn((sh1, sh2, sh3), |(a, b, c)| odd_func(a, b, c));
        let line = array_reshape::image_to_line(&arr.view());
        let arr2 = array_reshape::line_to_image(&line.view(), sh1, sh2);
        assert!(arr.eq(&arr2))
    }

    #[test]
    fn image_to_line_to_image_rectangle() {
        let (sh1, sh2, sh3) = (50usize, 80usize, 3usize);
        let arr: Array3<u8> = Array3::from_shape_fn((sh1, sh2, sh3), |(a, b, c)| odd_func(a, b, c));
        let line = array_reshape::image_to_line(&arr.view());
        let arr2 = array_reshape::line_to_image(&line.view(), sh1, sh2);
        assert!(arr.eq(&arr2))
    }
}
