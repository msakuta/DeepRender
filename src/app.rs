mod graph;
mod image;
mod ui_panel;

use eframe::egui::{
    self,
    plot::{Legend, Line, PlotPoints},
    widgets::plot::Plot,
    Ui,
};

use self::graph::NetworkRender;

use crate::{
    activation::ActivationFn,
    bg_image::BgImage,
    fit_model::{FitModel, ImageSize, IMAGE_HALFWIDTH},
    model::Model,
    optimizer::OptimizerType,
    sampler::{Sampler, TrainBatch},
};

pub struct DeepRenderApp {
    fit_model: FitModel,
    current_fit_model: FitModel,
    file_name: String,
    /// Image size used in synthesized images. FileImage should read size from file.
    synth_image_size: i32,
    sampler: Box<dyn Sampler>,
    train_batch: TrainBatch,
    batch_size: usize,
    image_size: Option<ImageSize>,
    hidden_layers: usize,
    hidden_nodes: usize,
    model: Model,
    rate: f64,
    trains_per_frame: usize,
    loss_history: Vec<f64>,
    weights_history: Vec<Vec<f64>>,
    activation_fn: ActivationFn,
    optimizer: OptimizerType,
    paused: bool,
    plot_network: bool,
    plot_weights: bool,
    print_weights: bool,
    angle: f64,
    upsample: usize,
    network_render: NetworkRender,

    // Widgets
    img: BgImage,
    img_predict: BgImage,
}

impl DeepRenderApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let fit_model = FitModel::Xor;
        let file_name = "alan.jpg".to_string();
        let (sampler, image_size) = fit_model.train_data(&file_name, IMAGE_HALFWIDTH).unwrap();
        let hidden_layers = 1;
        let mut arch = vec![fit_model.num_inputs()];
        for _ in 0..hidden_layers {
            arch.push(2);
        }
        arch.push(1);
        let activation_fn = ActivationFn::Sigmoid;
        let optimizer = OptimizerType::Steepest;
        let model = Model::new(&arch, activation_fn, optimizer.instantiate(&arch));
        Self {
            fit_model,
            current_fit_model: fit_model,
            file_name,
            synth_image_size: IMAGE_HALFWIDTH,
            sampler,
            train_batch: TrainBatch::Sequence,
            batch_size: 1,
            image_size,
            hidden_layers,
            hidden_nodes: 2,
            model,
            rate: 0.,
            trains_per_frame: 1,
            loss_history: vec![],
            weights_history: vec![],
            activation_fn,
            optimizer,
            paused: false,
            plot_network: true,
            plot_weights: true,
            print_weights: true,
            angle: 0.,
            upsample: 1,
            network_render: NetworkRender::Lines,
            img: BgImage::new(),
            img_predict: BgImage::new(),
        }
    }

    fn reset(&mut self) {
        self.current_fit_model = self.fit_model;
        (self.sampler, self.image_size) = self
            .fit_model
            .train_data(&self.file_name, self.synth_image_size)
            .unwrap();
        let mut arch = vec![self.fit_model.num_inputs()];
        for _ in 0..self.hidden_layers {
            arch.push(self.hidden_nodes);
        }
        arch.push(1);
        self.model = Model::new(&arch, self.activation_fn, self.optimizer.instantiate(&arch));
        self.loss_history = vec![];
        self.weights_history = vec![];
        self.img.clear();
    }

    fn learn_iter(&mut self) {
        let rate = (10.0f64).powf(self.rate);
        for _ in 0..self.trains_per_frame {
            let samples = self.sampler.sample(self.train_batch, self.batch_size);
            self.model.learn(rate, &samples);
        }
        self.loss_history.push(self.model.loss(self.sampler.full()));
        if self.plot_weights {
            self.add_weights_history();
        }
    }

    fn loss_history(&self) -> Line {
        let points: PlotPoints = self
            .loss_history
            .iter()
            .enumerate()
            .map(|(i, val)| [i as f64, *val])
            .collect();
        Line::new(points)
            .color(eframe::egui::Color32::from_rgb(100, 200, 100))
            .name("Loss")
    }

    fn add_weights_history(&mut self) {
        let elems = self.model.get_weights()[0].flat().len();
        if self.weights_history.len() <= elems {
            self.weights_history.resize(elems, vec![]);
        }
        for i in 0..elems {
            if let Some(weights_history) = self.weights_history.get_mut(i) {
                weights_history.push(self.model.get_weights()[0].flat()[i]);
            }
        }
    }

    fn weights_history(&self) -> Vec<Line> {
        self.weights_history
            .iter()
            .enumerate()
            .map(|(i, weights_history)| {
                let points: PlotPoints = weights_history
                    .iter()
                    .enumerate()
                    .map(|(t, v)| [t as f64, *v])
                    .collect();
                Line::new(points)
                    .color(eframe::egui::Color32::from_rgb(
                        (i % 2 * 200) as u8,
                        (i % 4 * 200) as u8,
                        (i % 8 * 100) as u8,
                    ))
                    .name(format!("weights[{}, {}]", i / 2, i % 2))
            })
            .collect()
    }

    fn func_plot(&self, ui: &mut Ui) {
        let plot = Plot::new("plot");
        plot.legend(Legend::default()).show(ui, |plot_ui| {
            let train = self.sampler.full();
            let points: PlotPoints = train
                .iter_rows()
                .map(|sample| [sample[0], sample[1]])
                .collect();
            let line = Line::new(points)
                .color(eframe::egui::Color32::from_rgb(0, 0, 255))
                .name("Training");
            plot_ui.line(line);
            let points: PlotPoints = train
                .iter_rows()
                .map(|sample| [sample[0], self.model.predict(sample)[(0, 0)]])
                .collect();
            let line_predict = Line::new(points).name("Predicted");
            plot_ui.line(line_predict);
        });
    }
}

impl eframe::App for DeepRenderApp {
    fn update(&mut self, ctx: &eframe::egui::Context, _frame: &mut eframe::Frame) {
        if !self.paused {
            ctx.request_repaint();

            self.learn_iter();
        }

        eframe::egui::SidePanel::right("side_panel")
            .min_width(200.)
            .show(ctx, |ui| self.ui_panel(ui));

        if self.plot_network {
            egui::TopBottomPanel::bottom("graph")
                .resizable(true)
                .min_height(100.)
                .show(ctx, |ui| {
                    self.paint_graph(ui);
                });
        }

        if self.plot_weights {
            egui::TopBottomPanel::bottom("weight_plot")
                .resizable(true)
                .min_height(100.)
                .default_height(125.)
                .show(ctx, |ui| {
                    let plot = Plot::new("plot");
                    plot.legend(Legend::default()).show(ui, |plot_ui| {
                        for line in self.weights_history() {
                            plot_ui.line(line)
                        }
                    });
                });
        }

        match self.sampler.full().cols() {
            2 => {
                egui::TopBottomPanel::bottom("func_plot")
                    .resizable(true)
                    .min_height(100.)
                    .default_height(125.)
                    .show(ctx, |ui| self.func_plot(ui));
            }
            3 | 4 => {
                egui::TopBottomPanel::bottom("image_plot")
                    .resizable(true)
                    .min_height(100.)
                    .default_height(125.)
                    .show(ctx, |ui| self.image_plot(ui));
            }
            _ => (),
        }

        eframe::egui::CentralPanel::default().show(ctx, |ui| {
            let plot = Plot::new("plot");
            plot.legend(Legend::default()).show(ui, |plot_ui| {
                plot_ui.line(self.loss_history());
            })
        });
    }
}
