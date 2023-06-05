use eframe::{
    egui::{
        self,
        plot::{Legend, Line, PlotPoints},
        widgets::plot::Plot,
        Frame, Ui,
    },
    epaint::{pos2, Color32, Pos2, Rect},
};

use crate::{activation::ActivationFn, fit_model::FitModel, matrix::Matrix, model::Model};

pub struct DeepRenderApp {
    fit_model: FitModel,
    train: Matrix,
    hidden_layers: usize,
    hidden_nodes: usize,
    model: Model,
    rate: f64,
    loss_history: Vec<f64>,
    weights_history: Vec<Vec<f64>>,
    activation_fn: ActivationFn,
}

impl DeepRenderApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let fit_model = FitModel::Xor;
        let train = fit_model.train_data();
        let hidden_layers = 1;
        let mut arch = vec![train.cols() - 1];
        for _ in 0..hidden_layers {
            arch.push(2);
        }
        arch.push(1);
        let activation_fn = ActivationFn::Sigmoid;
        let model = Model::new(
            &arch,
            activation_fn.get(),
            activation_fn.get_derive(),
            activation_fn.random_scale(),
        );
        Self {
            fit_model: FitModel::Xor,
            train,
            hidden_layers,
            hidden_nodes: 2,
            model,
            rate: 0.,
            loss_history: vec![],
            weights_history: vec![],
            activation_fn,
        }
    }

    fn reset(&mut self) {
        self.train = self.fit_model.train_data();
        let mut arch = vec![self.train.cols() - 1];
        for _ in 0..self.hidden_layers {
            arch.push(self.hidden_nodes);
        }
        arch.push(1);
        self.model = Model::new(
            &arch,
            self.activation_fn.get(),
            self.activation_fn.get_derive(),
            self.activation_fn.random_scale(),
        );
        self.loss_history = vec![];
        self.weights_history = vec![];
    }

    fn learn_iter(&mut self) {
        self.model.learn((10.0f64).powf(self.rate), &self.train);
        self.loss_history.push(self.model.loss(&self.train));
        self.add_weights_history();
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
        let elems = self.model.weights[0].flat().len();
        if self.weights_history.len() <= elems {
            self.weights_history.resize(elems, vec![]);
        }
        for i in 0..elems {
            if let Some(weights_history) = self.weights_history.get_mut(i) {
                weights_history.push(self.model.weights[0].flat()[i]);
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

    fn paint_graph(&self, ui: &mut Ui) {
        Frame::canvas(ui.style()).show(ui, |ui| {
            let (response, painter) =
                ui.allocate_painter(ui.available_size(), egui::Sense::hover());

            let to_screen = egui::emath::RectTransform::from_to(
                Rect::from_min_size(Pos2::ZERO, response.rect.size()),
                response.rect,
            );

            let to_color = |weights: &Matrix, i| {
                let weight = weights[(i, 0)];
                if weight < 0. {
                    Color32::from_rgb((weight.abs() * 255.).min(255.).max(0.) as u8, 0, 0)
                } else {
                    Color32::from_rgb(0, (weights[(i, 0)] * 255.).min(255.).max(0.) as u8, 0)
                }
            };

            for (n, weights) in self.model.weights.iter().enumerate() {
                let x = 30. + n as f32 * 70.;
                for i in 0..weights.rows() {
                    // let rect = Rect{ min: pos2(30., 30. + i as f32 * 30.), max: pos2(80., 50. + i as f32 * 30.) };
                    let center = pos2(x, 30. + i as f32 * 30.);
                    painter.circle(
                        to_screen.transform_pos(center),
                        10.,
                        to_color(weights, i),
                        (1., Color32::GRAY),
                    );
                }

                let center = pos2(x + 70., 30. + weights.cols() as f32 / 2.);
                painter.circle(
                    to_screen.transform_pos(center),
                    10.,
                    Color32::from_rgb(255, 0, 0),
                    (1., Color32::GRAY),
                );

                for i in 0..weights.rows() {
                    // let rect = Rect{ min: pos2(30., 30. + i as f32 * 30.), max: pos2(80., 50. + i as f32 * 30.) };
                    let soure = pos2(x, 30. + i as f32 * 30.);
                    for j in 0..self.model.arch[n + 1] {
                        let dest = pos2(x + 70., 30. + j as f32 * 30.);
                        painter.line_segment(
                            [
                                to_screen.transform_pos(soure),
                                to_screen.transform_pos(dest),
                            ],
                            (1., to_color(weights, i)),
                        );
                    }
                }
            }
        });
    }

    fn ui_panel(&mut self, ui: &mut Ui) {
        if ui.button("Reset").clicked() {
            self.reset();
        }

        ui.horizontal(|ui| {
            ui.label("Fit model:");
            ui.radio_value(&mut self.fit_model, FitModel::Xor, "Xor");
            ui.radio_value(&mut self.fit_model, FitModel::Sine, "Sine");
        });

        ui.horizontal(|ui| {
            ui.label("Activation fn:");
            ui.radio_value(&mut self.activation_fn, ActivationFn::Sigmoid, "Sigmoid");
            ui.radio_value(&mut self.activation_fn, ActivationFn::Relu, "ReLU");
            ui.radio_value(&mut self.activation_fn, ActivationFn::Silu, "SiLU");
        });

        ui.group(|ui| {
            ui.label("Architecture:");

            ui.horizontal(|ui| {
                ui.label("Hidden layers:");
                ui.add(egui::Slider::new(&mut self.hidden_layers, 1..=5));
            });

            ui.horizontal(|ui| {
                ui.label("Hidden nodes:");
                ui.add(egui::Slider::new(&mut self.hidden_nodes, 1..=10));
            });
        });

        ui.group(|ui| {
            ui.horizontal(|ui| {
                ui.label("Descent rate log10:");
                ui.add(egui::Slider::new(&mut self.rate, -10.0..=0.));
            });
            ui.label(format!("Descent rate: {}", (10.0f64).powf(self.rate)));
        });

        ui.label(format!("Loss: {}", self.model.loss(&self.train)));
        ui.label(format!("Model:\n{}", self.model));
        for sample in self.train.iter_rows() {
            let predict = self.model.predict(sample);
            ui.label(format!("{} -> {}", Matrix::new_row(&sample[0..2]), predict));
        }
    }

    fn func_plot(&self, ui: &mut Ui) {
        let plot = Plot::new("plot");
        plot.legend(Legend::default()).show(ui, |plot_ui| {
            let points: PlotPoints = self
                .train
                .iter_rows()
                .map(|sample| [sample[0], sample[1]])
                .collect();
            let line = Line::new(points)
                .color(eframe::egui::Color32::from_rgb(0, 0, 255))
                .name("Training");
            plot_ui.line(line);
            let points: PlotPoints = self
                .train
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
        ctx.request_repaint();

        self.learn_iter();

        eframe::egui::SidePanel::right("side_panel")
            .min_width(200.)
            .show(ctx, |ui| self.ui_panel(ui));

        egui::TopBottomPanel::bottom("graph")
            .resizable(true)
            .min_height(100.)
            .show(ctx, |ui| {
                self.paint_graph(ui);
            });

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

        if self.train.cols() == 2 {
            egui::TopBottomPanel::bottom("func_plot")
                .resizable(true)
                .min_height(100.)
                .default_height(125.)
                .show(ctx, |ui| self.func_plot(ui));
        }

        eframe::egui::CentralPanel::default().show(ctx, |ui| {
            let plot = Plot::new("plot");
            plot.legend(Legend::default()).show(ui, |plot_ui| {
                plot_ui.line(self.loss_history());
            })
        });
    }
}
