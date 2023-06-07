use std::fmt::Display;

use crate::{activation::ActivationFn, matrix::Matrix, optimizer::Optimizer};

pub(crate) struct Model {
    arch: Vec<usize>,
    weights: Vec<Matrix>,
    activation: fn(f64) -> f64,
    activation_derive: fn(f64) -> f64,
    optimizer: Box<dyn Optimizer>,
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
        activation_fn: ActivationFn,
        optimizer: Box<dyn Optimizer>,
    ) -> Self {
        Self {
            arch: shapes.to_vec(),
            weights: shapes_to_matrix(shapes, |(n, m)| {
                Matrix::rand(*n + 1, *m).scale(activation_fn.random_scale())
            }),
            activation: activation_fn.get(),
            activation_derive: activation_fn.get_derive(),
            optimizer,
        }
    }

    pub(crate) fn get_arch(&self) -> &[usize] {
        &self.arch
    }

    pub(crate) fn learn(&mut self, rate: f64, train: &Matrix) {
        self.learn_iter(rate, train);
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

    pub(crate) fn get_weights(&self) -> &[Matrix] {
        &self.weights
    }

    fn learn_iter(&mut self, rate: f64, samples: &Matrix) {
        struct LayerCache {
            signal: Matrix,
            interm: Matrix,
        }

        // Forward propagation
        let mut signal = samples.cols_range(0, samples.cols() - 1);
        let mut layer_caches = vec![];
        for weights in self.weights.iter() {
            let signal_biased = signal.hstack(&Matrix::ones(signal.rows(), 1));
            let interm = &signal_biased * &weights;
            signal = interm.map(self.activation);
            layer_caches.push(LayerCache {
                signal: signal_biased,
                interm,
            });
        }

        // Back propagation
        let mut loss = samples.col(samples.cols() - 1) - signal;

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
                    let mut diff_sum = 0.;
                    for l in 0..loss.rows() {
                        diff_sum += loss[(l, j)] * signal_biased[(l, i)] * interm_derived[(l, j)];
                    }
                    diff[(i, j)] = diff_sum;
                }
            }
            *weights += self
                .optimizer
                .apply(l, &diff)
                .scale(rate / samples.rows() as f64);
            let loss_derived = loss.elementwise_mul(&interm_derived);
            loss = &loss_derived * &weights.t().cols_range(0, weights.rows() - 1);
        }
    }
}

pub(crate) fn shapes_to_matrix(
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
