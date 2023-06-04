mod app;
mod matrix;
mod model;

use app::DeepRenderApp;

fn sigmoid(x: f64) -> f64 {
    1. / (1. + (-x).exp())
}

fn sigmoid_derive(x: f64) -> f64 {
    let sigmoid_x = sigmoid(x);
    (1. - sigmoid_x) * sigmoid_x
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
