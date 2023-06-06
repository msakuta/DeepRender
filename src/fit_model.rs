use crate::matrix::Matrix;

#[derive(PartialEq, Eq)]
pub(crate) enum FitModel {
    Xor,
    Sine,
    Image,
}

impl FitModel {
    pub(crate) fn train_data(&self) -> Matrix {
        match self {
            Self::Xor => {
                // let train = [[0., 0., 0.], [0., 1., 1.], [1., 0., 1.], [1., 1., 1.]];
                // let train = [[0., 0., 0.], [0., 1., 0.], [1., 0., 0.], [1., 1., 1.]];
                // let train = [[0., 0., 0.], [0., 1., 1.], [1., 0., 1.], [1., 1., 0.]];
                // let train = [[0., 0., 0.], [0., 1., 1.], [1., 0., 0.], [1., 1., 1.]];
                Matrix::new([[0., 0., 0.], [0., 1., 1.], [1., 0., 1.], [1., 1., 0.]])
            }
            Self::Sine => {
                let data: Vec<_> = (-20..=20)
                    .map(|f| [f as f64, (f as f64 / 4.).sin() * 0.5 + 0.5])
                    .collect();
                Matrix::from_slice(&data)
            }
            Self::Image => {
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
                Matrix::from_slice(&data)
            }
        }
    }
}
