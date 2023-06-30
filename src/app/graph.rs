use crate::{bg_image::BgImage, matrix::Matrix};

use super::DeepRenderApp;

use eframe::{
    egui::{self, Frame, Ui},
    epaint::{pos2, Color32, Pos2, Rect},
};

#[derive(PartialEq, Eq, Debug)]
pub(super) enum NetworkRender {
    Lines,
    Image,
}

impl DeepRenderApp {
    pub(super) fn paint_graph(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.label("Weights plot:");
            ui.radio_value(&mut self.network_render, NetworkRender::Lines, "Lines");
            ui.radio_value(&mut self.network_render, NetworkRender::Image, "Image");
        });

        const NODE_OFFSET: f32 = 30.;
        const NODE_RADIUS: f32 = 7.;
        const NODE_INTERVAL: f32 = NODE_RADIUS * 3.;
        const LAYER_INTERVAL: f32 = 70.;
        const PIXEL_SIZE: f32 = 5.;
        const IMAGE_OFFSET: f32 = 10. + NODE_RADIUS;

        Frame::canvas(ui.style()).show(ui, |ui| {
            let (response, painter) =
                ui.allocate_painter(ui.available_size(), egui::Sense::hover());

            let to_screen = egui::emath::RectTransform::from_to(
                Rect::from_min_size(Pos2::ZERO, response.rect.size()),
                response.rect,
            );

            let to_rgb = |weight: f64| {
                if weight < 0. {
                    [(weight.abs() * 255.).min(255.).max(0.) as u8, 0, 0]
                } else {
                    [0, (weight * 255.).min(255.).max(0.) as u8, 0]
                }
            };

            let to_color = |weight: f64| {
                let rgb = to_rgb(weight);
                Color32::from_rgb(rgb[0], rgb[1], rgb[2])
            };

            let mut x = NODE_OFFSET;
            let mut x_offsets = Vec::with_capacity(self.model.get_weights().len() + 1);
            x_offsets.push(x);

            for (n, weights) in self.model.get_weights().iter().enumerate() {
                match self.network_render {
                    NetworkRender::Lines => {
                        for i in 0..weights.rows() {
                            // let rect = Rect{ min: pos2(30., 30. + i as f32 * 30.), max: pos2(80., 50. + i as f32 * 30.) };
                            let soure = pos2(x, NODE_OFFSET + i as f32 * NODE_INTERVAL);
                            for j in 0..self.model.get_arch()[n + 1] {
                                let dest = pos2(
                                    x + LAYER_INTERVAL,
                                    NODE_OFFSET + j as f32 * NODE_INTERVAL,
                                );
                                painter.line_segment(
                                    [
                                        to_screen.transform_pos(soure),
                                        to_screen.transform_pos(dest),
                                    ],
                                    (1., to_color(weights[(i, j)])),
                                );
                            }
                        }
                        x += LAYER_INTERVAL;
                    }
                    NetworkRender::Image => {
                        let mut img = BgImage::new();
                        img.paint(
                            &response,
                            &painter,
                            weights,
                            |weights: &Matrix| {
                                let image = weights
                                    .flat()
                                    .iter()
                                    .copied()
                                    .map(to_rgb)
                                    .flatten()
                                    .collect::<Vec<_>>();
                                egui::ColorImage::from_rgb([weights.cols(), weights.rows()], &image)
                            },
                            [x + IMAGE_OFFSET, 25.],
                            PIXEL_SIZE,
                        );
                        x += weights.cols() as f32 * PIXEL_SIZE + IMAGE_OFFSET * 2.;
                    }
                }
                x_offsets.push(x);
            }

            let last_x = x_offsets.last().copied();

            for (weights, x) in self.model.get_weights().iter().zip(x_offsets.into_iter()) {
                for i in 0..weights.rows() {
                    let center = pos2(x, NODE_OFFSET + i as f32 * NODE_INTERVAL);
                    painter.circle(
                        to_screen.transform_pos(center),
                        NODE_RADIUS,
                        to_color(weights[(i, 0)]),
                        (1., Color32::GRAY),
                    );
                }
            }

            if let Some((x, rows)) = last_x.zip(self.model.get_arch().last()) {
                for y in 0..*rows {
                    painter.circle(
                        to_screen.transform_pos(pos2(x, NODE_OFFSET + y as f32 * NODE_INTERVAL)),
                        NODE_RADIUS,
                        Color32::GRAY,
                        (1., Color32::GRAY),
                    );
                }
            }
        });
    }
}
