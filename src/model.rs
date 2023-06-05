use std::fmt::Display;

use crate::{
    matrix::Matrix,
    optimizer::{Optimizer, OptimizerType},
};

pub(crate) struct Model {
    pub arch: Vec<usize>,
    pub weights: Vec<Matrix>,
    pub activation: fn(f64) -> f64,
    pub activation_derive: fn(f64) -> f64,
    optimizer: Optimizer,
}

impl Display for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Architecture: {:?}", self.arch)?;
        writeln!(f, "Weights: {:?}", self.weights)?;
        Ok(())
    }
}

impl Model {
    pub(crate) fn new(
        shapes: &[usize],
        activation: fn(f64) -> f64,
        activation_derive: fn(f64) -> f64,
        random_scale: f64,
        optimizer: OptimizerType,
    ) -> Self {
        fn shapes_to_matrix(
            shapes: &[usize],
            mat_create: impl Fn((&usize, &usize)) -> Matrix,
        ) -> Vec<Matrix> {
            shapes
                .iter()
                .take(shapes.len() - 1)
                .zip(shapes.iter().skip(1))
                .map(mat_create)
                .collect()
        }
        let optimizer = match optimizer {
            OptimizerType::Steepest => Optimizer::Steepest,
            OptimizerType::Adam => {
                let momentum = shapes_to_matrix(shapes, |(n, m)| Matrix::zeros(*n + 1, *m));
                Optimizer::Adam {
                    t: 0.,
                    momentum1: momentum.clone(),
                    momentum2: momentum,
                }
            }
        };
        Self {
            arch: shapes.to_vec(),
            weights: shapes_to_matrix(shapes, |(n, m)| {
                Matrix::rand(*n + 1, *m).scale(random_scale)
            }),
            activation,
            activation_derive,
            optimizer,
        }
    }

    pub(crate) fn loss(&self, train: &Matrix) -> f64 {
        train
            .iter_rows()
            .map(|sample| {
                let loss = sample[sample.len() - 1] - self.predict(sample)[(0, 0)];
                loss.powf(2.)
            })
            .fold(0., |acc, cur| acc + cur)
            / 2.
    }

    pub(crate) fn predict(&self, sample: &[f64]) -> Matrix {
        let mut input = Matrix::new_row(&sample[0..self.arch[0]]);
        for weights in &self.weights {
            let signal = input.hstack(&Matrix::ones(1, 1));
            let interm = &signal * &weights;
            input = interm.map(self.activation);
        }
        input
    }

    pub(crate) fn learn(&mut self, rate: f64, train: &Matrix) {
        for row in 0..train.rows() {
            self.learn_iter(rate, train.row(row));
        }
    }

    fn learn_iter(&mut self, rate: f64, sample: &[f64]) {
        let input = Matrix::new_row(&sample[0..sample.len() - 1]);

        struct LayerCache {
            signal: Matrix,
            interm: Matrix,
        }

        // Forward propagation
        let mut signal = input.clone();
        let mut layer_caches = vec![];
        for weights in self.weights.iter() {
            let signal_biased = signal.hstack(&Matrix::ones(1, 1));
            let interm = &signal_biased * &weights;
            signal = interm.map(self.activation);
            layer_caches.push(LayerCache {
                signal: signal_biased,
                interm,
            });
        }

        // Back propagation
        let mut loss = Matrix::new([[sample[sample.len() - 1] - signal[(0, 0)]]]);

        for ((l, layer_cache), weights) in layer_caches
            .iter()
            .enumerate()
            .zip(self.weights.iter_mut())
            .rev()
        {
            let interm_derived = layer_cache.interm.map(self.activation_derive);
            let weights_shape = weights.shape();
            let signal_biased = &layer_cache.signal;
            let mut diff = Matrix::zeros(weights_shape.0, weights_shape.1);
            for i in 0..weights_shape.0 {
                for j in 0..weights_shape.1 {
                    diff[(i, j)] = loss[(0, j)] * signal_biased[(0, i)] * interm_derived[(0, j)];
                }
            }
            *weights += self.optimizer.apply(l, &diff).scale(rate);
            loss = &(&loss.t() * &interm_derived).sum_col() * &weights.t();
        }
    }
}
