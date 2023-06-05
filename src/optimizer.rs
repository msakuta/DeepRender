use crate::{matrix::Matrix, model::shapes_to_matrix};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OptimizerType {
    /// Steepest gradient descent. Simple, stupid, unstable.
    Steepest,
    /// Adaptive momentum. Diederik P. Kingma and Jimmy Lei Ba, https://doi.org/10.48550/arXiv.1412.6980
    Adam,
}

pub(crate) trait Optimizer {
    fn new(shapes: &[usize]) -> Self;
    fn apply(&mut self, l: usize, diff: &Matrix) -> Matrix;
}

pub(crate) struct SteepestDescentOptimizer;

impl Optimizer for SteepestDescentOptimizer {
    fn new(_shapes: &[usize]) -> Self {
        Self
    }

    fn apply(&mut self, _l: usize, diff: &Matrix) -> Matrix {
        diff.clone()
    }
}

pub(crate) struct AdamOptimizer {
    t: f64,
    momentum1: Vec<Matrix>,
    momentum2: Vec<Matrix>,
}

const BETA1: f64 = 0.9;
const BETA2: f64 = 0.999;
const EPSILON: f64 = 1e-8;

impl Optimizer for AdamOptimizer {
    fn new(shapes: &[usize]) -> Self {
        let momentum = shapes_to_matrix(shapes, |(n, m)| Matrix::zeros(*n + 1, *m));
        Self {
            t: 0.,
            momentum1: momentum.clone(),
            momentum2: momentum,
        }
    }

    fn apply(&mut self, l: usize, diff: &Matrix) -> Matrix {
        self.t += 1.;
        let momentum1 = &mut self.momentum1[l];
        for (m, d) in momentum1.zip_mut(diff) {
            *m = BETA1 * *m + (1. - BETA1) * *d
        }
        let momentum2 = &mut self.momentum2[l];
        for (m, d) in momentum2.zip_mut(diff) {
            *m = BETA2 * *m + (1. - BETA2) * d.powf(2.)
        }
        let mut momentum1hat = momentum1.scale(1. / (1. - BETA1.powf(self.t)));
        let momentum2hat = momentum2.scale(1. / (1. - BETA2.powf(self.t)));
        for (m1, m2) in momentum1hat.zip_mut(&momentum2hat) {
            *m1 /= m2.sqrt() + EPSILON;
        }
        momentum1hat
    }
}
