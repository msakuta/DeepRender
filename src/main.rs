mod activation;
mod app;
mod bg_image;
mod fit_model;
mod matrix;
mod model;
mod optimizer;
mod sampler;

use app::DeepRenderApp;

#[cfg(not(target_arch = "wasm32"))]
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

// when compiling to web using trunk.
#[cfg(target_arch = "wasm32")]
fn main() {
    // Make sure panics are logged using `console.error`.
    console_error_panic_hook::set_once();

    // Redirect tracing to console.log and friends:
    tracing_wasm::set_as_global_default();

    let mut web_options = eframe::WebOptions::default();

    // We insist to use dark theme, because light theme looks dumb.
    web_options.follow_system_theme = false;
    web_options.default_theme = eframe::Theme::Dark;

    wasm_bindgen_futures::spawn_local(async {
        eframe::start_web(
            "the_canvas_id", // hardcode it
            web_options,
            Box::new(|cc| Box::new(DeepRenderApp::new(cc))),
        )
        .await
        .expect("failed to start eframe");
    });
}
