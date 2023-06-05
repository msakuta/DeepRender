fn sigmoid(x: f64) -> f64 {
    1. / (1. + (-x).exp())
}

fn sigmoid_derive(x: f64) -> f64 {
    let sigmoid_x = sigmoid(x);
    (1. - sigmoid_x) * sigmoid_x
}

fn relu(x: f64) -> f64 {
    x.max(0.)
}

fn relu_derive(x: f64) -> f64 {
    if x < 0. {
        0.
    } else {
        1.
    }
}

fn silu(x: f64) -> f64 {
    x * sigmoid(x)
}

fn silu_derive(x: f64) -> f64 {
    let sigmoid_x = sigmoid(x);
    sigmoid_x * (1. + (1. - sigmoid_x))
}

#[derive(PartialEq, Eq)]
pub(crate) enum ActivationFn {
    Sigmoid,
    Relu,
    Silu,
}

impl ActivationFn {
    pub(crate) fn get(&self) -> fn(f64) -> f64 {
        match self {
            Self::Sigmoid => sigmoid,
            Self::Relu => relu,
            Self::Silu => silu,
        }
    }

    pub(crate) fn get_derive(&self) -> fn(f64) -> f64 {
        match self {
            Self::Sigmoid => sigmoid_derive,
            Self::Relu => relu_derive,
            Self::Silu => silu_derive,
        }
    }

    pub(crate) fn random_scale(&self) -> f64 {
        match self {
            Self::Sigmoid => 1.,
            Self::Relu => 1.,
            Self::Silu => 1.,
        }
    }
}
