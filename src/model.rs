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
            / 2.
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

    pub(crate) fn learn<const N: usize>(&mut self, rate: f64, train: &[[f64; N]]) {
        for sample in train {
            self.learn_iter(rate, sample);
        }
    }

    fn learn_once<const N: usize>(&mut self, rate: f64, sample: &[f64; N]) {
        let input = Matrix::new_row(&sample[0..N - 1]);
        let signal = input.clone();

        let weights = &self.weights[0];
        let signal_biased = signal.hstack(&Matrix::ones(1, 1));
        let interm1 = &signal_biased * weights;
        // println!("signal: {:?}, weights: {:?}, shape: {:?}", signal_biased.shape(), weights.shape(), interm1.shape());
        let signal1 = interm1.map(sigmoid);
        drop(weights);

        let weights = &self.weights[1];
        let signal1_biased = signal1.hstack(&Matrix::ones(1, 1));
        let interm2 = &signal1_biased * weights;
        // println!("signal: {:?}, weights: {:?}, shape: {:?}", signal_biased.shape(), weights.shape(), interm1.shape());
        let signal2 = interm2.map(sigmoid);
        drop(weights);
        // dbg!(&signals);

        // Back propagation
        let mut loss = Matrix::new([[sample[N - 1] - signal2[(0, 0)]]]);
        let interm2_derived = interm2.map(sigmoid_derive);
        // println!("weights: {:?}, interm2_derived: {:?}, signal: {:?}", weights.shape(), interm2_derived.shape(), signal.shape());
        let weights = &mut self.weights[1];
        // println!("weights: {:?}, interm2_derived: {:?}, signal: {:?}", weights.shape(), interm2_derived.shape(), signal.shape());
        // println!("{weights} * {interm2_derived} = {diff}");
        // let signal1_biased = signal1.hstack(&Matrix::ones(1, 1));
        // println!("weights: {weights} signal1_biased: {signal1_biased}, diff: {diff}, interm2_derived: {interm2_derived}, loss: {loss}");
        for i in 0..weights.rows() {
            for j in 0..weights.cols() {
                weights[(i, j)] +=
                    rate * loss[(0, j)] * signal1_biased[(0, i)] * interm2_derived[(0, 0)];
            }
        }
        drop(weights);

        let loss1 = (&loss * &interm2_derived).sum_col();
        let interm1_derived = interm1.map(sigmoid_derive);
        let weights = &mut self.weights[0];
        // println!("weights: {:?}, interm1_derived: {:?}, loss1: {:?}, signal: {:?}", weights.shape(), interm1_derived.shape(), loss1.shape(), signal.shape());
        // let signal_biased = signal.hstack(&Matrix::ones(1, 1));
        for i in 0..weights.rows() {
            for j in 0..weights.cols() {
                weights[(i, j)] +=
                    rate * loss1[(0, 0)] * signal_biased[(0, i)] * interm1_derived[(0, 0)];
            }
        }
    }

    fn learn_iter<const N: usize>(&mut self, rate: f64, sample: &[f64; N]) {
        let input = Matrix::new_row(&sample[0..N - 1]);

        // Forward propagation
        let mut signal = input.clone();
        let mut signals = vec![];
        let mut interms = vec![];
        for weights in &self.weights {
            let signal_biased = signal.hstack(&Matrix::ones(1, 1));
            let interm = &signal_biased * &weights;
            // println!("signal: {:?}, weights: {:?}, shape: {:?}", signal_biased.shape(), weights.shape(), interm.shape());
            signal = interm.map(sigmoid);
            signals.push(signal_biased);
            interms.push(interm);
        }

        // Back propagation
        let mut loss = Matrix::new([[sample[N - 1] - signal[(0, 0)]]]);
        for ((interm, signal), weights) in interms
            .iter()
            .rev()
            .zip(signals.iter().rev())
            .zip(self.weights.iter_mut().rev())
        {
            let interm_derived = interm.map(sigmoid_derive);
            // println!("weights: {:?}, interm_derived: {:?}, signal: {:?}", weights.shape(), interm_derived.shape(), signal.shape());
            // let diff = weights as &Matrix * &interm_derived.t();
            // println!("weights: {:?}, interm_derived: {:?}, signal: {:?}, loss: {:?}", weights.shape(), interm_derived.shape(), signal.shape(), loss.shape());
            // println!("{weights} * {interm_derived} = {diff}");
            let input_biased = signal; //.hstack(&Matrix::ones(1, 1));
                                       // println!("weights: {weights} input_biased: {input_biased}, diff: {diff}, interm_derived: {interm_derived}, loss: {loss}");
            for i in 0..weights.rows() {
                for j in 0..weights.cols() {
                    weights[(i, j)] +=
                        rate * loss[(0, 0)] * input_biased[(0, i)] * interm_derived[(0, 0)];
                }
            }
            loss = (&loss.t() * &interm_derived).sum_col();
        }
    }
}
