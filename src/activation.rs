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

#[derive(PartialEq, Eq)]
pub(crate) enum ActivationFn {
    Sigmoid,
    Relu,
}

impl ActivationFn {
    pub(crate) fn get(&self) -> fn(f64) -> f64 {
        match self {
            Self::Sigmoid => sigmoid,
            Self::Relu => relu,
        }
    }

    pub(crate) fn get_derive(&self) -> fn(f64) -> f64 {
        match self {
            Self::Sigmoid => sigmoid_derive,
            Self::Relu => relu_derive,
        }
    }
}
