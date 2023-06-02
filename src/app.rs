use eframe::{
    egui::{
        self,
        plot::{Line, PlotPoints},
        widgets::plot::Plot,
        Frame, Ui,
    },
    epaint::{pos2, Color32, Pos2, Rect},
};

use crate::{matrix::Matrix, model::Model};

pub struct DeepRenderApp {
    train: Vec<[f64; 3]>,
    model: Model,
    rate: f64,
    loss_history: Vec<f64>,
}

impl DeepRenderApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let train = [[0., 0., 0.], [0., 1., 1.], [1., 0., 1.], [1., 1., 1.]];
        // let train = [[0., 0., 0.], [0., 1., 0.], [1., 0., 0.], [1., 1., 1.]];
        // let train = [[0., 0., 0.], [0., 1., 1.], [1., 0., 1.], [1., 1., 0.]];
        Self {
            train: train.to_vec(),
            model: Model::new(&[3, 1]),
            rate: 1.,
            loss_history: vec![],
        }
    }

    fn learn_iter(&mut self) {
        self.model.learn(self.rate, &self.train);
        self.loss_history.push(self.model.loss(&self.train));
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
            .name("circle")
    }

    fn paint_graph(&self, ui: &mut Ui) {
        Frame::canvas(ui.style()).show(ui, |ui| {
            let (response, painter) =
                ui.allocate_painter(ui.available_size(), egui::Sense::hover());

            let to_screen = egui::emath::RectTransform::from_to(
                Rect::from_min_size(Pos2::ZERO, response.rect.size()),
                response.rect,
            );

            for (n, weights) in self.model.weights.iter().enumerate() {
                for i in 0..weights.rows() {
                    // let rect = Rect{ min: pos2(30., 30. + i as f32 * 30.), max: pos2(80., 50. + i as f32 * 30.) };
                    let center = pos2(30. + n as f32 * 70., 30. + i as f32 * 30.);
                    painter.circle(
                        to_screen.transform_pos(center),
                        10.,
                        Color32::from_rgb((weights[(i, 0)] * 255.).min(255.).max(0.) as u8, 0, 0),
                        (1., Color32::GRAY),
                    );
                }

                let center = pos2(100. + n as f32 * 70., 30. + weights.cols() as f32 / 2.);
                painter.circle(
                    to_screen.transform_pos(center),
                    10.,
                    Color32::from_rgb(255, 0, 0),
                    (1., Color32::GRAY),
                );

                for i in 0..weights.rows() {
                    // let rect = Rect{ min: pos2(30., 30. + i as f32 * 30.), max: pos2(80., 50. + i as f32 * 30.) };
                    let soure = pos2(30., 30. + i as f32 * 30.);
                    let dest = pos2(100., 30.);
                    painter.line_segment(
                        [
                            to_screen.transform_pos(soure),
                            to_screen.transform_pos(dest),
                        ],
                        (
                            1.,
                            Color32::from_rgb(
                                (weights[(i, 0)] * 255.).min(255.).max(0.) as u8,
                                0,
                                0,
                            ),
                        ),
                    );
                }
            }
        });
    }
}

impl eframe::App for DeepRenderApp {
    fn update(&mut self, ctx: &eframe::egui::Context, _frame: &mut eframe::Frame) {
        ctx.request_repaint();

        self.learn_iter();

        eframe::egui::SidePanel::right("side_panel")
            .min_width(200.)
            .show(ctx, |ui| {
                ui.label(format!("Loss: {}", self.model.loss(&self.train)));
                ui.label(format!("Model:\n{}", self.model));
                for sample in &self.train {
                    let predict = self.model.predict(sample);
                    ui.label(format!("{} -> {}", Matrix::new_row(&sample[0..2]), predict));
                }
            });

        egui::TopBottomPanel::bottom("graph")
            .resizable(true)
            .min_height(100.)
            .show(ctx, |ui| {
                self.paint_graph(ui);
            });

        eframe::egui::CentralPanel::default().show(ctx, |ui| {
            let plot = Plot::new("plot");
            plot.show(ui, |plot_ui| {
                plot_ui.line(self.loss_history());
            })
        });
    }
}
