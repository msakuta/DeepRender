use crate::matrix::Matrix;
use ray_rust::{
    quat::Quat,
    render::{
        render, RenderColor, RenderEnv, RenderFloor, RenderMaterial, RenderObject, RenderPattern,
        RenderSphere,
    },
    vec3::Vec3,
};

use image::io::Reader as ImageReader;
use std::{collections::HashMap, path::Path, sync::Arc};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum FitModel {
    Xor,
    Sine,
    /// Synthetic image
    SynthImage,
    /// Image loaded from file
    FileImage,
    /// A synthesized image by ray tracing renderer
    RaytraceImage,
    /// A synthesized image by ray tracing renderer
    Raytrace3D,
}

pub(crate) const IMAGE_HALFWIDTH: i32 = 15;
pub(crate) const ANGLES: i32 = 10;

pub(crate) type ImageSize = [usize; 2];

impl FitModel {
    pub(crate) fn train_data(
        &self,
        file_name: &impl AsRef<Path>,
        image_size: i32,
    ) -> Result<(Matrix, Option<ImageSize>), Box<dyn std::error::Error>> {
        match self {
            Self::Xor => {
                // let train = [[0., 0., 0.], [0., 1., 1.], [1., 0., 1.], [1., 1., 1.]];
                // let train = [[0., 0., 0.], [0., 1., 0.], [1., 0., 0.], [1., 1., 1.]];
                let train = [[0., 0., 0.], [0., 1., 1.], [1., 0., 1.], [1., 1., 0.]];
                // let train = [[0., 0., 0.], [0., 1., 1.], [1., 0., 0.], [1., 1., 1.]];
                Ok((Matrix::new(train), None))
            }
            Self::Sine => {
                let data: Vec<_> = (-50..=50)
                    .map(|f| [f as f64 / 50., (f as f64 / 4.).sin() * 0.5 + 0.5])
                    .collect();
                Ok((Matrix::from_slice(&data), None))
            }
            Self::SynthImage => {
                let image_size_i = image_size as i32;
                let image_width = image_size as usize * 2 + 1;
                let data: Vec<_> = (-image_size_i..=image_size_i)
                    .map(|y| {
                        (-image_size_i..=image_size_i).map(move |x| {
                            [
                                x as f64 / image_size as f64 - 1.,
                                y as f64 / image_size as f64 - 1.,
                                (x as f64 / 4.).sin() * (y as f64 / 4.).sin() * 0.5 + 0.5,
                            ]
                        })
                    })
                    .flatten()
                    .collect();
                Ok((Matrix::from_slice(&data), Some([image_width; 2])))
            }
            Self::FileImage => {
                let img = ImageReader::open(file_name)?.decode()?.into_luma8();
                let width = img.width();
                let height = img.height();
                let fwidth = width as f64;
                let fheight = height as f64;
                let data: Vec<_> = img
                    .enumerate_pixels()
                    .map(|(x, y, px)| {
                        [
                            x as f64 / fwidth - 0.5,
                            y as f64 / fheight - 0.5,
                            px.0[0] as f64 / 255.,
                        ]
                    })
                    .collect();
                Ok((
                    Matrix::from_slice(&data),
                    Some([width as usize, height as usize]),
                ))
            }
            Self::RaytraceImage => {
                let image_size_i = image_size as i32;
                let image_width = image_size as usize * 2 + 1;
                let buf = &render_main(image_width);
                let data: Vec<_> = (-image_size_i..=image_size_i)
                    .map(|y| {
                        (-image_size_i..=image_size_i).map(move |x| {
                            let ux = (x + image_size_i) as usize;
                            let uy = (y + image_size_i) as usize;
                            [
                                x as f64 / image_size_i as f64 - 1.,
                                y as f64 / image_size_i as f64 - 1.,
                                // (x as f64 / 4.).sin() * (y as f64 / 4.).sin() * 0.5 + 0.5,
                                buf[uy * image_width + ux as usize] as f64,
                            ]
                        })
                    })
                    .flatten()
                    .collect();
                Ok((Matrix::from_slice(&data), Some([image_width; 2])))
            }
            Self::Raytrace3D => {
                let image_size_u = image_size as usize;
                let buf = &render3d_main(image_size_u, ANGLES as usize);
                let data: Vec<_> = (0..ANGLES)
                    .map(|angle| {
                        let angle_offset = angle as usize * image_size_u * image_size_u;
                        (0..image_size_u)
                            .map(move |y| {
                                (0..image_size_u).map(move |x| {
                                    [
                                        x as f64 / image_size as f64 - 0.5,
                                        y as f64 / image_size as f64 - 0.5,
                                        angle as f64 / ANGLES as f64 - 0.5,
                                        // (x as f64 / 4.).sin() * (y as f64 / 4.).sin() * 0.5 + 0.5,
                                        buf[angle_offset + y * image_size_u + x] as f64,
                                    ]
                                })
                            })
                            .flatten()
                    })
                    .flatten()
                    .collect();
                Ok((Matrix::from_slice(&data), Some([image_size_u; 2])))
            }
        }
    }
}

fn render_main(image_width: usize) -> Vec<f32> {
    let mut materials: HashMap<String, Arc<RenderMaterial>> = HashMap::new();

    fn bg(_env: &RenderEnv, _pos: &Vec3) -> RenderColor {
        RenderColor::new(0.5, 0.5, 0.5)
    }

    let floor_material = Arc::new(
        RenderMaterial::new(
            "floor".to_string(),
            RenderColor::new(0.75, 1.0, 0.0),
            RenderColor::new(0.0, 0.0, 0.0),
            0,
            0.,
            0.0,
        )
        .pattern(RenderPattern::Solid)
        .pattern_scale(300.)
        .pattern_angle_scale(0.2)
        .texture_ok("bar.png"),
    );
    materials.insert("floor".to_string(), floor_material);

    let red_material = Arc::new(
        RenderMaterial::new(
            "red".to_string(),
            RenderColor::new(0.8, 0.0, 0.0),
            RenderColor::new(0.0, 0.0, 0.0),
            24,
            0.,
            0.0,
        )
        .glow_dist(5.),
    );

    // let transparent_material = Arc::new(
    //     RenderMaterial::new(
    //         "transparent".to_string(),
    //         RenderColor::new(0.0, 0.0, 0.0),
    //         RenderColor::new(0.0, 0.0, 0.0),
    //         0,
    //         1.,
    //         1.5,
    //     )
    //     .frac(RenderColor::new(1.49998, 1.49999, 1.5)),
    // );

    let objects: Vec<RenderObject> = vec![
        /* Plane */
        RenderObject::Floor(RenderFloor::new_raw(
            materials.get("floor").unwrap().clone(),
            Vec3::new(0.0, -150.0, 0.0),
            Vec3::new(0., 1., 0.),
        )),
        // RenderFloor::new (floor_material,       Vec3::new(-300.0,   0.0,  0.0),  Vec3::new(1., 0., 0.)),
        /* Spheres */
        // RenderSphere::new(mirror_material.clone(), 80.0, Vec3::new(0.0, -30.0, 172.0)),
        // RenderSphere::new(mirror_material, 80.0, Vec3::new(-200.0, -30.0, 172.0)),
        // RenderSphere::new(red_material, 80.0, Vec3::new(-200.0, -200.0, 172.0)),
        /*	{80.0F,  70.0F,-200.0F,150.0F, 0.0F, 0.0F, 0.8F, 0.0F, 0.0F, 0.0F, 0.0F,24, 1., 1., {1.}},*/
        RenderSphere::new(red_material, 100.0, Vec3::new(0.0, -50.0, 100.0)),
        /*	{000.F, 0.F, 0.F, 1500.F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 1.0F, 1.0F,24, 0, 0},*/
        /*	{100.F, -70.F, -150.F, 160.F, 0.0F, 0.5F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,24, .5F, .2F},*/
    ];

    use std::f32::consts::PI;

    let render_env = RenderEnv::new(
        Vec3::new(0., 2., -100.),
        Vec3::new(PI / 12., -PI / 2., -PI / 2.), /* pyr */
        image_width as i32,
        image_width as i32,
        1.,
        1.,
        bg,
    )
    .materials(materials)
    .objects(objects)
    .light(Vec3::new(50., 60., -50.));
    let mut buf = vec![0.; image_width * image_width];
    render(
        &render_env,
        &mut |x, y, color| buf[y as usize * image_width + x as usize] = color.r,
        1,
    );
    buf
}

fn render3d_main(image_width: usize, angles: usize) -> Vec<f32> {
    let mut materials: HashMap<String, Arc<RenderMaterial>> = HashMap::new();

    fn bg(_env: &RenderEnv, _pos: &Vec3) -> RenderColor {
        RenderColor::new(0.5, 0.5, 0.5)
    }

    let floor_material = Arc::new(
        RenderMaterial::new(
            "floor".to_string(),
            RenderColor::new(0.75, 1.0, 0.0),
            RenderColor::new(0.0, 0.0, 0.0),
            0,
            0.,
            0.0,
        )
        .pattern(RenderPattern::Solid)
        .pattern_scale(300.)
        .pattern_angle_scale(0.2)
        .texture_ok("bar.png"),
    );
    materials.insert("floor".to_string(), floor_material);

    let red_material = Arc::new(
        RenderMaterial::new(
            "red".to_string(),
            RenderColor::new(0.8, 0.0, 0.0),
            RenderColor::new(0.0, 0.0, 0.0),
            24,
            0.,
            0.0,
        )
        .glow_dist(5.),
    );

    // let transparent_material = Arc::new(
    //     RenderMaterial::new(
    //         "transparent".to_string(),
    //         RenderColor::new(0.0, 0.0, 0.0),
    //         RenderColor::new(0.0, 0.0, 0.0),
    //         0,
    //         1.,
    //         1.5,
    //     )
    //     .frac(RenderColor::new(1.49998, 1.49999, 1.5)),
    // );

    let objects: Vec<RenderObject> = vec![
        /* Plane */
        RenderObject::Floor(RenderFloor::new_raw(
            materials.get("floor").unwrap().clone(),
            Vec3::new(0.0, -150.0, 0.0),
            Vec3::new(0., 1., 0.),
        )),
        // RenderFloor::new (floor_material,       Vec3::new(-300.0,   0.0,  0.0),  Vec3::new(1., 0., 0.)),
        /* Spheres */
        // RenderSphere::new(mirror_material.clone(), 80.0, Vec3::new(0.0, -30.0, 172.0)),
        // RenderSphere::new(mirror_material, 80.0, Vec3::new(-200.0, -30.0, 172.0)),
        // RenderSphere::new(red_material, 80.0, Vec3::new(-200.0, -200.0, 172.0)),
        /*	{80.0F,  70.0F,-200.0F,150.0F, 0.0F, 0.0F, 0.8F, 0.0F, 0.0F, 0.0F, 0.0F,24, 1., 1., {1.}},*/
        RenderSphere::new(red_material, 100.0, Vec3::new(0.0, -50.0, 0.0)),
        /*	{000.F, 0.F, 0.F, 1500.F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 1.0F, 1.0F,24, 0, 0},*/
        /*	{100.F, -70.F, -150.F, 160.F, 0.0F, 0.5F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,24, .5F, .2F},*/
    ];

    use std::f32::consts::PI;

    let mut render_env = RenderEnv::new(
        Vec3::new(0., 2., -150.),
        Vec3::new(PI / 12., -PI / 2., -PI / 2.), /* pyr */
        image_width as i32,
        image_width as i32,
        1.,
        1.,
        bg,
    )
    .materials(materials)
    .objects(objects)
    .light(Vec3::new(50., 60., -50.));
    let angle_stride = image_width * image_width;
    let mut buf = vec![0.; angle_stride * image_width * image_width];
    for angle in 0..angles {
        let angle_f = angle as f32 / angles as f32;
        let angle_rad = angle_f * PI / 2.;
        render_env.camera.position.x = angle_rad.sin() * 200.;
        render_env.camera.position.z = -angle_rad.cos() * 200.;
        render_env.camera.pyr.y = (-angle_f - 1.) * PI / 2.;
        render_env.camera.rotation = Quat::from_pyr(&render_env.camera.pyr);
        render(
            &render_env,
            &mut |x, y, color| {
                buf[angle * angle_stride + y as usize * image_width + x as usize] = color.r
            },
            1,
        );
    }
    buf
}
