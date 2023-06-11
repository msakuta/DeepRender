use rand::{seq::SliceRandom, Rng};
use ray_rust::{
    quat::Quat,
    render::{render, RenderEnv},
    vec3::Vec3,
};

use crate::{
    fit_model::{angle_to_camera, render_scene},
    matrix::Matrix,
};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum TrainBatch {
    Sequence,
    Shuffle,
    Full,
}

pub(crate) trait Sampler {
    fn sample(&mut self, train_batch: TrainBatch, batch_size: usize) -> Matrix;
    fn full(&self) -> &Matrix;
}

pub(crate) struct MatrixSampler {
    train: Matrix,
    order: Vec<usize>,
}

impl Sampler for MatrixSampler {
    fn sample(&mut self, train_batch: TrainBatch, batch_size: usize) -> Matrix {
        match train_batch {
            TrainBatch::Sequence => {
                if self.order.is_empty() {
                    self.order = (0..self.train.rows()).collect();
                }
                let samples = batch_size.min(self.order.len());
                let mut train = Matrix::zeros(samples, self.train.cols());
                for j in 0..samples {
                    train
                        .row_mut(j)
                        .copy_from_slice(self.train.row(self.order.pop().unwrap()));
                }
                train
            }
            TrainBatch::Shuffle => {
                if self.order.is_empty() {
                    self.order = (0..self.train.rows()).collect();
                    self.order.shuffle(&mut rand::thread_rng());
                }
                let samples = batch_size.min(self.order.len());
                let mut train = Matrix::zeros(samples, self.train.cols());
                for j in 0..samples {
                    train
                        .row_mut(j)
                        .copy_from_slice(self.train.row(self.order.pop().unwrap()));
                }
                train
            }
            TrainBatch::Full => self.train.clone(),
        }
    }

    fn full(&self) -> &Matrix {
        &self.train
    }
}

impl MatrixSampler {
    pub(crate) fn new(train: Matrix) -> Self {
        Self {
            train,
            order: vec![],
        }
    }
}

pub(crate) struct RaytraceSampler {
    train: Matrix,

    // Renderer params
    // materials: HashMap<String, Arc<RenderMaterial>>,
    render_env: RenderEnv,
}

impl Sampler for RaytraceSampler {
    fn sample(&mut self, _train_batch: TrainBatch, batch_size: usize) -> Matrix {
        let mut rng = rand::thread_rng();
        use std::f32::consts::PI;

        let mut samples = Matrix::zeros(batch_size, 4);
        for i in 0..batch_size {
            let angle = rand::random::<f32>();
            let angle_f = angle;
            let (yaw, x, y) = angle_to_camera(angle_f);
            self.render_env.camera.position.x = x;
            self.render_env.camera.position.z = y;
            self.render_env.camera.pyr.y = yaw;
            let ray_x = rng.gen::<f32>() - 0.5;
            let ray_y = rng.gen::<f32>() - 0.5;
            self.render_env.camera.rotation = Quat::from_pyr(&self.render_env.camera.pyr)
                * Quat::from_pyr(&Vec3::new(ray_y * PI / 2., ray_x * PI / 2., 0.));
            samples[(i, 0)] = ray_x as f64;
            samples[(i, 1)] = ray_y as f64;
            samples[(i, 2)] = angle as f64 - 0.5;
            render(
                &self.render_env,
                &mut |_x, _y, color| samples[(i, 3)] = color.r as f64,
                1,
            );
        }
        // println!("RaytraceSample: {samples:?}");
        samples
    }

    fn full(&self) -> &Matrix {
        &self.train
    }
}

impl RaytraceSampler {
    pub(crate) fn new(train: Matrix) -> Self {
        Self {
            train,
            render_env: render_scene(1),
        }
    }
}
