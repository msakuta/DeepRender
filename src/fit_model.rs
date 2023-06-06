use crate::matrix::Matrix;

use image::io::Reader as ImageReader;
use std::io::Cursor;

#[derive(PartialEq, Eq)]
pub(crate) enum FitModel {
    Xor,
    Sine,
    /// Synthetic image
    SynthImage,
    /// Image loaded from file
    FileImage,
}

impl FitModel {
    pub(crate) fn train_data(&self) -> Result<Matrix, Box<dyn std::error::Error>> {
        match self {
            Self::Xor => {
                // let train = [[0., 0., 0.], [0., 1., 1.], [1., 0., 1.], [1., 1., 1.]];
                // let train = [[0., 0., 0.], [0., 1., 0.], [1., 0., 0.], [1., 1., 1.]];
                let train = [[0., 0., 0.], [0., 1., 1.], [1., 0., 1.], [1., 1., 0.]];
                // let train = [[0., 0., 0.], [0., 1., 1.], [1., 0., 0.], [1., 1., 1.]];
                Ok(Matrix::new(train))
            }
            Self::Sine => {
                let data: Vec<_> = (-20..=20)
                    .map(|f| [f as f64, (f as f64 / 4.).sin() * 0.5 + 0.5])
                    .collect();
                Ok(Matrix::from_slice(&data))
            }
            Self::SynthImage => {
                let data: Vec<_> = (-10..=10)
                    .map(|y| {
                        (-10..=10).map(move |x| {
                            [
                                x as f64,
                                y as f64,
                                (x as f64 / 4.).sin() * (y as f64 / 4.).sin() * 0.5 + 0.5,
                            ]
                        })
                    })
                    .flatten()
                    .collect();
                Ok(Matrix::from_slice(&data))
            }
            Self::FileImage => {
                let img = ImageReader::open("alan.jpg")?.decode()?.into_luma8();
                let width = img.width() as f64;
                let height = img.height() as f64;
                let data: Vec<_> = img
                    .enumerate_pixels()
                    .map(|(x, y, px)| {
                        [
                            (x as f64 - width / 2.) / 2.,
                            (y as f64 - height / 2.) / 2.,
                            px.0[0] as f64 / 255.,
                        ]
                    })
                    .collect();
                Ok(Matrix::from_slice(&data))
            }
        }
    }
}
