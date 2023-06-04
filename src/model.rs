use std::fmt::Display;

use crate::matrix::Matrix;

pub(crate) struct Model {
    pub arch: Vec<usize>,
    pub weights: Vec<Matrix>,
    pub activation: fn(f64) -> f64,
    pub activation_derive: fn(f64) -> f64,
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
    ) -> Self {
        Self {
            arch: shapes.to_vec(),
            weights: shapes
                .iter()
                .take(shapes.len() - 1)
                .zip(shapes.iter().skip(1))
                .map(|(n, m)| Matrix::rand(*n + 1, *m))
                .collect(),
            activation,
            activation_derive,
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
            input = interm.map(self.activation);
        }
        input
    }

    pub(crate) fn learn<const N: usize>(&mut self, rate: f64, train: &[[f64; N]]) {
        for sample in train {
            self.learn_iter(rate, sample);
        }
    }

    #[allow(dead_code)]
    fn learn_once<const N: usize>(&mut self, rate: f64, sample: &[f64; N]) {
        let input = Matrix::new_row(&sample[0..N - 1]);
        let signal = input.clone();

        let weights = &self.weights[0];
        let signal_biased = signal.hstack(&Matrix::ones(1, 1));
        let interm1 = &signal_biased * weights;
        // println!("signal: {:?}, weights: {:?}, shape: {:?}", signal_biased.shape(), weights.shape(), interm1.shape());
        let signal1 = interm1.map(self.activation);
        drop(weights);

        let weights = &self.weights[1];
        let signal1_biased = signal1.hstack(&Matrix::ones(1, 1));
        let interm2 = &signal1_biased * weights;
        // println!("signal: {:?}, weights: {:?}, shape: {:?}", signal_biased.shape(), weights.shape(), interm1.shape());
        let signal2 = interm2.map(self.activation);
        drop(weights);
        // dbg!(&signals);

        // Back propagation
        let loss = Matrix::new([[sample[N - 1] - signal2[(0, 0)]]]);
        let interm2_derived = interm2.map(self.activation_derive);
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

        let loss1 = &(&loss * &interm2_derived).sum_col() * &self.weights[1].t();
        let interm1_derived = interm1.map(self.activation_derive);
        let weights_shape = self.weights[0].shape();
        // println!("weights1: {:?}, weights2: {:?}, interm1_derived: {:?}, loss1: {:?}, signal: {:?}",
        //     weights_shape, self.weights[1].shape(), interm1_derived.shape(), loss1.shape(), signal.shape());
        // let signal_biased = signal.hstack(&Matrix::ones(1, 1));
        for i in 0..weights_shape.0 {
            for j in 0..weights_shape.1 {
                self.weights[0][(i, j)] +=
                    rate * loss1[(0, j)] * signal_biased[(0, i)] * interm1_derived[(0, j)];
            }
        }
    }

    fn learn_iter<const N: usize>(&mut self, rate: f64, sample: &[f64; N]) {
        let input = Matrix::new_row(&sample[0..N - 1]);

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
            // println!("signal: {:?}, weights: {:?}, shape: {:?}", signal_biased.shape(), weights.shape(), interm.shape());
            signal = interm.map(self.activation);
            layer_caches.push(LayerCache {
                signal: signal_biased,
                interm,
            });
        }

        // Back propagation
        let mut loss = Matrix::new([[sample[N - 1] - signal[(0, 0)]]]);
        // let last_layer = layer_caches.last().unwrap();
        // let interm2_derived = last_layer.interm.map(sigmoid_derive);
        // // println!("weights: {:?}, interm2_derived: {:?}, signal: {:?}", weights.shape(), interm2_derived.shape(), signal.shape());
        // let weights = self.weights.last_mut().unwrap();
        // // println!("weights: {:?}, interm2_derived: {:?}, signal: {:?}", weights.shape(), interm2_derived.shape(), signal.shape());
        // // println!("{weights} * {interm2_derived} = {diff}");
        // let signal1_biased = &last_layer.signal;
        // // println!("weights: {weights} signal1_biased: {signal1_biased}, diff: {diff}, interm2_derived: {interm2_derived}, loss: {loss}");
        // for i in 0..weights.rows() {
        //     for j in 0..weights.cols() {
        //         weights[(i, j)] +=
        //             rate * loss[(0, j)] * signal1_biased[(0, i)] * interm2_derived[(0, 0)];
        //     }
        // }
        // drop(weights);

        // let mut loss = (&loss * &interm2_derived).sum_col();
        for ((_l, layer_cache), weights) in layer_caches
            .iter()
            .enumerate()
            .zip(self.weights.iter_mut())
            .rev()
        {
            let interm_derived = layer_cache.interm.map(self.activation_derive);
            // println!("weights: {:?}, interm_derived: {:?}, signal: {:?}", weights.shape(), interm_derived.shape(), signal.shape());
            // let diff = weights as &Matrix * &interm_derived.t();
            // println!("{weights} * {interm_derived} = {diff}");
            let weights_shape = weights.shape();
            // println!(
            //     "layer {l}: weights: {:?}, interm_derived: {:?}, signal: {:?}, loss: {:?}",
            //     weights_shape,
            //     interm_derived.shape(),
            //     layer_cache.signal.shape(),
            //     loss.shape()
            // );
            let signal_biased = &layer_cache.signal;
            // println!("weights: {weights} signal_biased: {signal_biased}, diff: {diff}, interm_derived: {interm_derived}, loss: {loss}");
            for i in 0..weights_shape.0 {
                for j in 0..weights_shape.1 {
                    weights[(i, j)] +=
                        rate * loss[(0, j)] * signal_biased[(0, i)] * interm_derived[(0, j)];
                }
            }
            loss = &(&loss.t() * &interm_derived).sum_col() * &weights.t();
        }
    }
}
