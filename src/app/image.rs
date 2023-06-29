use crate::{
    fit_model::{FitModel, ANGLES},
    matrix::Matrix,
    model::Model,
};

use super::DeepRenderApp;

use eframe::egui::{self, Frame, Ui};

impl DeepRenderApp {
    pub(super) fn image_plot(&mut self, ui: &mut Ui) {
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
                        &self.model,
                        |model: &Model| {
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
