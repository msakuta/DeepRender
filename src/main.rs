mod activation;
mod app;
mod bg_image;
mod fit_model;
mod matrix;
mod model;
mod optimizer;

use app::DeepRenderApp;

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
