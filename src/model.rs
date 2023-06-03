use std::fmt::Display;

use crate::{matrix::Matrix, sigmoid, sigmoid_derive};

pub(crate) struct Model {
    pub arch: Vec<usize>,
    pub weights: Vec<Matrix>,
}

impl Display for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Architecture: {:?}", self.arch)?;
        writeln!(f, "Weights: {:?}", self.weights)?;
        Ok(())
    }
}

impl Model {
    pub(crate) fn new(shapes: &[usize]) -> Self {
        Self {
            arch: shapes.to_vec(),
            weights: shapes
                .iter()
                .take(shapes.len() - 1)
                .zip(shapes.iter().skip(1))
                .map(|(n, m)| Matrix::rand(*n + 1, *m))
                .collect(),
        }
    }

    pub(crate) fn loss(&self, train: &[[f64; 3]]) -> f64 {
        train
            .iter()
            .map(|sample| {
                let loss = sample[2] - self.predict(sample)[(0, 0)];
                loss.powf(2.)
            })
            .fold(0., |acc, cur| acc + cur)
    }

    pub(crate) fn predict(&self, sample: &[f64; 3]) -> Matrix {
        let mut input = Matrix::new_row(&sample[0..2]);
        for weights in &self.weights {
            let signal = input.hstack(&Matrix::ones(1, 1));
            let interm = &signal * &weights;
            input = interm.map(sigmoid);
        }
        input
    }

    pub(crate) fn learn(&mut self, rate: f64, train: &[[f64; 3]]) {
        for sample in train {
            let input = Matrix::new_row(&sample[0..2]);
            let mut signal = input.clone();
            let mut interm_opt = None;
            for weights in &self.weights {
                let signal_biased = signal.hstack(&Matrix::ones(1, 1));
                // println!("signal: {:?}, weights: {:?}", signal_biased.shape(), weights.shape());
                let interm = &signal_biased * &weights;
                signal = interm.map(sigmoid);
                interm_opt = Some(interm);
            }
            let loss = sample[2] - signal[(0, 0)];
            let derive = interm_opt.unwrap().map(sigmoid_derive)[(0, 0)];
            // println!("{input} * {model} = {predict}, loss = {loss} derive = {derive}");
            let input_biased = input.hstack(&Matrix::ones(1, 1));
            for i in 0..self.weights[0].rows() {
                self.weights[0][(i, 0)] += rate * loss * input_biased[(0, i)] * derive;
            }
        }
    }
}
