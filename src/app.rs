use eframe::egui::{
    plot::{Line, PlotPoints},
    widgets::plot::Plot,
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
        let train = [[0., 0., 0.], [0., 1., 1.], [1., 0., 1.], [1., 1., 1.]];
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
}

impl eframe::App for DeepRenderApp {
    fn update(&mut self, ctx: &eframe::egui::Context, frame: &mut eframe::Frame) {
        ctx.request_repaint();

        self.learn_iter();

        eframe::egui::SidePanel::right("side_panel").show(ctx, |ui| {
            ui.label(format!("Loss: {}", loss_fn(&self.train, &self.model)));
            ui.label(format!("Model: {}", self.model));
        });

        eframe::egui::CentralPanel::default().show(ctx, |ui| {
            let plot = eframe::egui::plot::Plot::new("plot");
            plot.show(ui, |plot_ui| {
                plot_ui.line(self.loss_history());
            })
        });
    }
}
