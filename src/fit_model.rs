use crate::matrix::Matrix;

use image::io::Reader as ImageReader;
use std::path::Path;

#[derive(PartialEq, Eq)]
pub(crate) enum FitModel {
    Xor,
    Sine,
    /// Synthetic image
    SynthImage,
    /// Image loaded from file
    FileImage,
}

pub(crate) type ImageSize = [usize; 2];

impl FitModel {
    pub(crate) fn train_data(
        &self,
        file_name: &impl AsRef<Path>,
    ) -> Result<(Matrix, Option<ImageSize>), Box<dyn std::error::Error>> {
        match self {
            Self::Xor => {
                // let train = [[0., 0., 0.], [0., 1., 1.], [1., 0., 1.], [1., 1., 1.]];
                // let train = [[0., 0., 0.], [0., 1., 0.], [1., 0., 0.], [1., 1., 1.]];
                let train = [[0., 0., 0.], [0., 1., 1.], [1., 0., 1.], [1., 1., 0.]];
                // let train = [[0., 0., 0.], [0., 1., 1.], [1., 0., 0.], [1., 1., 1.]];
                Ok((Matrix::new(train), None))
            }
            Self::Sine => {
                let data: Vec<_> = (-20..=20)
                    .map(|f| [f as f64, (f as f64 / 4.).sin() * 0.5 + 0.5])
                    .collect();
                Ok((Matrix::from_slice(&data), None))
            }
            Self::SynthImage => {
                const IMAGE_HALFWIDTH: i32 = 10;
                const IMAGE_WIDTH: usize = IMAGE_HALFWIDTH as usize * 2 + 1;
                let data: Vec<_> = (-IMAGE_HALFWIDTH..=IMAGE_HALFWIDTH)
                    .map(|y| {
                        (-IMAGE_HALFWIDTH..=IMAGE_HALFWIDTH).map(move |x| {
                            [
                                x as f64,
                                y as f64,
                                (x as f64 / 4.).sin() * (y as f64 / 4.).sin() * 0.5 + 0.5,
                            ]
                        })
                    })
                    .flatten()
                    .collect();
                Ok((Matrix::from_slice(&data), Some([IMAGE_WIDTH; 2])))
            }
            Self::FileImage => {
                let img = ImageReader::open(file_name)?.decode()?.into_luma8();
                let width = img.width();
                let height = img.height();
                let fwidth = width as f64;
                let fheight = height as f64;
                let data: Vec<_> = img
                    .enumerate_pixels()
                    .map(|(x, y, px)| {
                        [
                            x as f64 / fwidth - 0.5,
                            y as f64 / fheight - 0.5,
                            px.0[0] as f64 / 255.,
                        ]
                    })
                    .collect();
                Ok((
                    Matrix::from_slice(&data),
                    Some([width as usize, height as usize]),
                ))
            }
        }
    }
}
