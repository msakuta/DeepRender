use eframe::{
    egui::{
        self,
        plot::{Line, PlotPoints},
        widgets::plot::Plot,
        Frame, Ui,
    },
    epaint::{pos2, Color32, Pos2, Rect},
};

use crate::{loss_fn, matrix::Matrix, sigmoid, sigmoid_derive};

pub struct DeepRenderApp {
    train: Vec<[f64; 3]>,
    model: Matrix,
    rate: f64,
    loss_history: Vec<f64>,
}

impl DeepRenderApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        // let train = [[0., 0., 0.], [0., 1., 1.], [1., 0., 1.], [1., 1., 1.]];
        // let train = [[0., 0., 0.], [0., 1., 0.], [1., 0., 0.], [1., 1., 1.]];
        let train = [[0., 0., 0.], [0., 1., 1.], [1., 0., 1.], [1., 1., 0.]];
        let model = Matrix::new([[0.67, 0.95, 0.32]]);
        Self {
            train: train.to_vec(),
            model,
            rate: 1.,
            loss_history: vec![],
        }
    }

    fn learn_iter(&mut self) {
        for sample in &self.train {
            let input = Matrix::new_row(&sample[0..2]).hstack(&Matrix::ones(1, 1));
            let interm = &input * &self.model.t();
            let predict = interm.map(sigmoid);
            let loss = sample[2] - predict[(0, 0)];
            let derive = interm.map(sigmoid_derive)[(0, 0)];
            // println!("{input} * {model} = {predict}, loss = {loss} derive = {derive}");
            self.model[(0, 0)] += self.rate * loss * input[(0, 0)] * derive;
            self.model[(0, 1)] += self.rate * loss * input[(0, 1)] * derive;
            self.model[(0, 2)] += self.rate * loss * input[(0, 2)] * derive;
        }
        self.loss_history.push(loss_fn(&self.train, &self.model));
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

            for i in 0..self.model.cols() {
                // let rect = Rect{ min: pos2(30., 30. + i as f32 * 30.), max: pos2(80., 50. + i as f32 * 30.) };
                let center = pos2(30., 30. + i as f32 * 30.);
                painter.circle(
                    to_screen.transform_pos(center),
                    10.,
                    Color32::from_rgb((self.model[(0, i)] * 255.).min(255.).max(0.) as u8, 0, 0),
                    (1., Color32::GRAY),
                );
            }

            let center = pos2(100., 30. + self.model.cols() as f32 / 2.);
            painter.circle(
                to_screen.transform_pos(center),
                10.,
                Color32::from_rgb(255, 0, 0),
                (1., Color32::GRAY),
            );

            for i in 0..self.model.cols() {
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
                            (self.model[(0, i)] * 255.).min(255.).max(0.) as u8,
                            0,
                            0,
                        ),
                    ),
                );
            }
        });
    }
}

impl eframe::App for DeepRenderApp {
    fn update(&mut self, ctx: &eframe::egui::Context, frame: &mut eframe::Frame) {
        ctx.request_repaint();

        self.learn_iter();

        eframe::egui::SidePanel::right("side_panel")
            .min_width(200.)
            .show(ctx, |ui| {
                ui.label(format!("Loss: {}", loss_fn(&self.train, &self.model)));
                ui.label(format!("Model: {}", self.model));
                for sample in &self.train {
                    let input = Matrix::new_row(&sample[0..2]).hstack(&Matrix::ones(1, 1));
                    let interm = &input * &self.model.t();
                    let predict = interm.map(sigmoid);
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
