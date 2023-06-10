use eframe::{
    egui::{
        self,
        plot::{Legend, Line, PlotPoints},
        widgets::plot::Plot,
        Frame, TextEdit, Ui,
    },
    epaint::{pos2, Color32, Pos2, Rect},
};

use crate::{
    activation::ActivationFn,
    bg_image::BgImage,
    fit_model::{FitModel, ImageSize, ANGLES, IMAGE_HALFWIDTH},
    matrix::Matrix,
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

            for (n, weights) in self.model.get_weights().iter().enumerate() {
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
                    for j in 0..self.model.get_arch()[n + 1] {
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
        ui.horizontal(|ui| {
            if ui.button("Reset").clicked() {
                self.reset();
            }

            let paused_label = if self.paused { "Unpause" } else { "Pause" };
            if ui.button(paused_label).clicked() {
                self.paused = !self.paused;
            }
        });

        ui.group(|ui| {
            ui.label("Fit model:");

            ui.horizontal(|ui| {
                ui.radio_value(&mut self.fit_model, FitModel::Xor, "Xor");
                ui.radio_value(&mut self.fit_model, FitModel::Sine, "Sine");
                ui.radio_value(&mut self.fit_model, FitModel::SynthImage, "SynthImage");
                ui.radio_value(&mut self.fit_model, FitModel::FileImage, "FileImage");
            });

            ui.horizontal(|ui| {
                ui.label("File name:");
                ui.add_enabled(
                    matches!(self.fit_model, FitModel::FileImage),
                    TextEdit::singleline(&mut self.file_name),
                );
            });

            ui.radio_value(
                &mut self.fit_model,
                FitModel::RaytraceImage,
                "RaytraceImage",
            );
            ui.radio_value(&mut self.fit_model, FitModel::Raytrace3D, "Raytrace3D");

            ui.horizontal(|ui| {
                ui.label("Image size:");
                ui.add_enabled(
                    matches!(
                        self.fit_model,
                        FitModel::RaytraceImage | FitModel::Raytrace3D
                    ),
                    egui::widgets::Slider::new(&mut self.synth_image_size, 8..=30),
                );
            })
        });

        ui.horizontal(|ui| {
            ui.label("Activation fn:");
            ui.radio_value(&mut self.activation_fn, ActivationFn::Sigmoid, "Sigmoid");
            ui.radio_value(&mut self.activation_fn, ActivationFn::Relu, "ReLU");
            ui.radio_value(&mut self.activation_fn, ActivationFn::Silu, "SiLU");
            ui.radio_value(&mut self.activation_fn, ActivationFn::Sin, "Sin");
        });

        ui.horizontal(|ui| {
            ui.label("Optimizer:");
            ui.radio_value(&mut self.optimizer, OptimizerType::Steepest, "Steepest");
            ui.radio_value(&mut self.optimizer, OptimizerType::Adam, "Adam");
        });

        ui.group(|ui| {
            ui.horizontal(|ui| {
                ui.label("Train batch: ");
                ui.radio_value(&mut self.train_batch, TrainBatch::Sequence, "Sequence");
                ui.radio_value(&mut self.train_batch, TrainBatch::Shuffle, "Shuffle");
                ui.radio_value(&mut self.train_batch, TrainBatch::Full, "Full");
            });

            ui.horizontal(|ui| {
                ui.label("Batch size:");
                // There is no real point having more than 50 batches.
                let max_batches = 50;
                ui.add_enabled(
                    !matches!(self.train_batch, TrainBatch::Full),
                    egui::Slider::new(&mut self.batch_size, 1..=max_batches),
                );
            })
        });

        ui.group(|ui| {
            ui.label("Architecture:");

            ui.horizontal(|ui| {
                ui.label("Hidden layers:");
                ui.add(egui::Slider::new(&mut self.hidden_layers, 1..=10));
            });

            ui.horizontal(|ui| {
                ui.label("Hidden nodes:");
                ui.add(egui::Slider::new(&mut self.hidden_nodes, 1..=30));
            });
        });

        ui.group(|ui| {
            ui.horizontal(|ui| {
                ui.label("Descent rate log10:");
                ui.add(egui::Slider::new(&mut self.rate, -10.0..=0.));
            });
            ui.label(format!("Descent rate: {}", (10.0f64).powf(self.rate)));
            ui.horizontal(|ui| {
                ui.label("Trains per frame:");
                ui.add(egui::widgets::Slider::new(
                    &mut self.trains_per_frame,
                    1..=150,
                ));
            });
        });

        ui.checkbox(&mut self.plot_network, "Plot network");

        ui.checkbox(&mut self.plot_weights, "Plot weights (uncheck for speed)");

        ui.checkbox(&mut self.print_weights, "Print weights (uncheck for speed)");

        ui.label(format!(
            "Loss: {}",
            self.loss_history.last().copied().unwrap_or(0.)
        ));

        if self.print_weights {
            ui.label(format!("Model:\n{}", self.model));
            for sample in self.sampler.full().iter_rows() {
                let predict = self.model.predict(sample);
                ui.label(format!("{} -> {}", Matrix::new_row(&sample[0..2]), predict));
            }
        }
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

    fn image_plot(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.label("Angle:");
            if ui
                .add(egui::widgets::Slider::new(
                    &mut self.angle,
                    (0.)..=(ANGLES - 1) as f64,
                ))
                .changed()
            {
                self.img.clear();
            }

            ui.label("Upsample:");
            ui.add(egui::widgets::Slider::new(&mut self.upsample, 1..=4));
        });

        Frame::canvas(ui.style()).show(ui, |ui| {
            let (response, painter) =
                ui.allocate_painter(ui.available_size(), egui::Sense::hover());

            let Some(image_size) = self.image_size else {
                return;
            };

            match self.current_fit_model {
                FitModel::Raytrace3D => {
                    let angle_stride = image_size[0] * image_size[1];
                    let angle = self.angle as usize;
                    self.img.paint(
                        &response,
                        &painter,
                        self.sampler.full(),
                        |train: &Matrix| {
                            let image = (0..angle_stride)
                                .map(|i| {
                                    let b = (train[(angle * angle_stride + i, 3)] * 255.)
                                        .max(0.)
                                        .min(255.);
                                    [b as u8; 3]
                                })
                                .flatten()
                                .collect::<Vec<_>>();
                            egui::ColorImage::from_rgb(image_size, &image)
                        },
                        [25., 25.],
                        5.,
                    );

                    let image_upsize =
                        [image_size[0] * self.upsample, image_size[1] * self.upsample];

                    self.img_predict.clear();
                    self.img_predict.paint(
                        &response,
                        &painter,
                        (self.sampler.full(), &self.model),
                        |(train, model): (&Matrix, &Model)| {
                            let image = (0..angle_stride * self.upsample * self.upsample)
                                .map(|i| {
                                    let x =
                                        (i % image_upsize[0]) as f64 / image_upsize[0] as f64 - 0.5;
                                    let y =
                                        (i / image_upsize[0]) as f64 / image_upsize[1] as f64 - 0.5;
                                    let sample = [x, y, self.angle / ANGLES as f64 - 0.5];
                                    let predict = model.predict(&sample);
                                    [(predict[(0, 0)] * 255.).max(0.).min(255.) as u8; 3]
                                })
                                .flatten()
                                .collect::<Vec<_>>();
                            egui::ColorImage::from_rgb(image_upsize, &image)
                        },
                        [image_size[0] as f32 * 5. + 50., 25.],
                        5. / self.upsample as f32,
                    );
                }
                _ => {
                    self.img.paint(
                        &response,
                        &painter,
                        self.sampler.full(),
                        |train: &Matrix| {
                            let image = (0..train.rows())
                                .map(|i| {
                                    [(train.flat()[i * train.cols() + 2] * 255.)
                                        .max(0.)
                                        .min(255.) as u8; 3]
                                })
                                .flatten()
                                .collect::<Vec<_>>();
                            egui::ColorImage::from_rgb(image_size, &image)
                        },
                        [25., 25.],
                        5.,
                    );

                    self.img_predict.clear();
                    self.img_predict.paint(
                        &response,
                        &painter,
                        (self.sampler.full(), &self.model),
                        |(train, model): (&Matrix, &Model)| {
                            let image = (0..train.rows())
                                .map(|i| {
                                    let sample = train.row(i);
                                    let predict = model.predict(sample);
                                    [(predict[(0, 0)] * 255.).max(0.).min(255.) as u8; 3]
                                })
                                .flatten()
                                .collect::<Vec<_>>();
                            egui::ColorImage::from_rgb(image_size, &image)
                        },
                        [image_size[0] as f32 * 5. + 50., 25.],
                        5.,
                    );
                }
            }
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
