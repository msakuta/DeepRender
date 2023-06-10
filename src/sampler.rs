use rand::seq::SliceRandom;

use crate::matrix::Matrix;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum TrainBatch {
    Sequence,
    Shuffle,
    Full,
}

pub(crate) trait Sampler {
    fn sample(&mut self, train_batch: TrainBatch, batch_size: usize) -> Matrix;
    fn full(&self) -> &Matrix;
}

pub(crate) struct MatrixSampler {
    train: Matrix,
    order: Vec<usize>,
}

impl Sampler for MatrixSampler {
    fn sample(&mut self, train_batch: TrainBatch, batch_size: usize) -> Matrix {
        match train_batch {
            TrainBatch::Sequence => {
                if self.order.is_empty() {
                    self.order = (0..self.train.rows()).collect();
                }
                let samples = batch_size.min(self.order.len());
                let mut train = Matrix::zeros(samples, self.train.cols());
                for j in 0..samples {
                    train
                        .row_mut(j)
                        .copy_from_slice(self.train.row(self.order.pop().unwrap()));
                }
                train
            }
            TrainBatch::Shuffle => {
                if self.order.is_empty() {
                    self.order = (0..self.train.rows()).collect();
                    self.order.shuffle(&mut rand::thread_rng());
                }
                let samples = batch_size.min(self.order.len());
                let mut train = Matrix::zeros(samples, self.train.cols());
                for j in 0..samples {
                    train
                        .row_mut(j)
                        .copy_from_slice(self.train.row(self.order.pop().unwrap()));
                }
                train
            }
            TrainBatch::Full => self.train.clone(),
        }
    }

    fn full(&self) -> &Matrix {
        &self.train
    }
}

impl MatrixSampler {
    pub(crate) fn new(train: Matrix) -> Self {
        Self {
            train,
            order: vec![],
        }
    }
}
