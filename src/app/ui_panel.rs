use eframe::egui::{self, TextEdit, Ui};

use super::DeepRenderApp;

use crate::{
    activation::ActivationFn, fit_model::FitModel, matrix::Matrix, optimizer::OptimizerType,
    sampler::TrainBatch,
};

impl DeepRenderApp {
    pub(super) fn ui_panel(&mut self, ui: &mut Ui) {
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
                #[cfg(not(target_arch = "wasm32"))]
                ui.radio_value(&mut self.fit_model, FitModel::FileImage, "FileImage");
            });

            #[cfg(not(target_arch = "wasm32"))]
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
                    1..=1000,
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
}
