use crate::matrix::Matrix;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OptimizerType {
    /// Steepest gradient descent. Simple, stupid, unstable.
    Steepest,
    /// Adaptive momentum. Diederik P. Kingma and Jimmy Lei Ba, https://doi.org/10.48550/arXiv.1412.6980
    Adam,
}

pub(crate) enum Optimizer {
    Steepest,
    Adam {
        t: f64,
        momentum1: Vec<Matrix>,
        momentum2: Vec<Matrix>,
    },
}

const BETA1: f64 = 0.9;
const BETA2: f64 = 0.999;
const EPSILON: f64 = 1e-8;

impl Optimizer {
    pub(crate) fn apply(&mut self, l: usize, diff: &Matrix) -> Matrix {
        match self {
            Self::Steepest => diff.clone(),
            Self::Adam {
                t,
                momentum1,
                momentum2,
            } => {
                *t += 1.;
                let momentum1 = &mut momentum1[l];
                for (m, d) in momentum1.zip_mut(diff) {
                    *m = BETA1 * *m + (1. - BETA1) * *d
                }
                let momentum2 = &mut momentum2[l];
                for (m, d) in momentum2.zip_mut(diff) {
                    *m = BETA2 * *m + (1. - BETA2) * d.powf(2.)
                }
                let mut momentum1hat = momentum1.scale(1. / (1. - BETA1.powf(*t)));
                let momentum2hat = momentum2.scale(1. / (1. - BETA2.powf(*t)));
                for (m1, m2) in momentum1hat.zip_mut(&momentum2hat) {
                    *m1 /= m2.sqrt() + EPSILON;
                }
                momentum1hat
            }
        }
    }
}
