mod app;
mod matrix;
mod model;

use app::DeepRenderApp;
use model::Model;

use self::matrix::Matrix;

fn sigmoid(x: f64) -> f64 {
    1. / (1. + (-x).exp())
}

fn sigmoid_derive(x: f64) -> f64 {
    let sigmoid_x = sigmoid(x);
    (1. - sigmoid_x) * sigmoid_x
}

fn loss_fn(train: &[[f64; 3]], model: &Matrix) -> f64 {
    train
        .iter()
        .map(|sample| {
            let input = Matrix::new_row(&sample[0..2]).hstack(&Matrix::ones(1, 1));
            let predict = (&input * &model.t()).map(sigmoid);
            let loss = sample[2] - predict[(0, 0)];
            loss.powf(2.)
        })
        .fold(0., |acc, cur| acc + cur)
}

fn learn(train: &[[f64; 3]]) {
    let mut model = Matrix::new([[0.67, 0.95, 0.32]]);

    println!("initial loss: {}", loss_fn(&train, &model));

    let rate = 1.;

    for i in 0..1000 {
        for sample in train {
            let input = Matrix::new_row(&sample[0..2]).hstack(&Matrix::ones(1, 1));
            let interm = &input * &model.t();
            let predict = interm.map(sigmoid);
            let loss = sample[2] - predict[(0, 0)];
            let derive = interm.map(sigmoid_derive)[(0, 0)];
            // println!("{input} * {model} = {predict}, loss = {loss} derive = {derive}");
            model[(0, 0)] += rate * loss * input[(0, 0)] * derive;
            model[(0, 1)] += rate * loss * input[(0, 1)] * derive;
            model[(0, 2)] += rate * loss * input[(0, 2)] * derive;
        }

        if i % 10 == 0 {
            println!("learned loss: {}, model: {model}", loss_fn(&train, &model));
        }
    }

    for sample in train {
        let input = Matrix::new_row(&sample[0..2]).hstack(&Matrix::ones(1, 1));
        let predict = (&input * &model.t()).map(sigmoid);
        let loss = sample[2] - predict[(0, 0)];
        let derive = sigmoid_derive(loss);
        println!("{input} * {model} = {predict}, loss = {loss} derive = {derive}");
    }

    // for x in -10..10 {
    //     println!("sigmoid({x}) = {}, d sigmoid / d x ({x}) = {}", sigmoid(x as f64), sigmoid_derive(x as f64));
    // }
}

fn main() {
    // Log to stdout (if you run with `RUST_LOG=debug`).
    // tracing_subscriber::fmt::init();

    let native_options = eframe::NativeOptions::default();
    eframe::run_native(
        "DeepRender",
        native_options,
        Box::new(|cc| Box::new(DeepRenderApp::new(cc))),
    )
    .unwrap();

    // let train = [[0., 0., 0.], [0., 1., 1.], [1., 0., 1.], [1., 1., 1.]];
    // let train = [[0., 0., 0.], [0., 1., 0.], [1., 0., 0.], [1., 1., 1.]];
    // learn(&train);
}
