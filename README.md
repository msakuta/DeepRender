# DeepRender

An experimental Neural Network trainer/visualizer in Rust

Try it on your browser! https://msakuta.github.io/DeepRender/

## Training on a function

A neural network is a universal function approximator.
Therefore, you can fit it to any function, including sine wave, given enough neurons.

![](images/screenshot.png)

Training process:

![](https://msakuta.github.io/images/showcase/DeepRender.gif)

## An example of training on an image

You can train the network to imitate an image!

![](images/screenshot02.png)


## An example of training 3D renderer

Training process:

![Training process](https://msakuta.github.io/images/showcase/DeepRender3DTrain.gif)

Result:

![Result](https://msakuta.github.io/images/showcase/DeepRender3DResult.gif)


# What's this?

This project attempts to implement a neural network trainer and visualizer with a GUI using eframe/egui without using any of the deep learning libraries.
Even the matrix operations are implemented from scratch.
As such, this project is not expected to work efficiently.

There are 4 models to train:

* XOR logical gate
* sinusoidal function
* Synthetic image (2D function field)
* An image
* A 3D scene rendered with ray tracing renderer ([ray-rust](https://github.com/msakuta/ray-rust))

You can switch the model, activation functions, the network architecture and the descent rate in real time.

See [this document](https://github.com/msakuta/typst-test/blob/gh-pages/neural-network.pdf) for more details about the theoretical background.

This project is inspired by this video series, where deep learning framework is implemented in plain C (if you can do it in C, why not in Rust?):
https://youtu.be/PGSba51aRYU


# How to build

Install [Rust](https://www.rust-lang.org/).

    cargo r

## How to build Wasm version

You can build the application to WebAssembly and run on the browser.

* Install [wasm-pack](https://rustwasm.github.io/wasm-pack/)
* Install [trunk](https://github.com/thedodd/trunk) by `cargo install trunk`
* Run `trunk serve` for development server, or
* Run `trunk build --release` for release build in `eframe/dist`

Note that you cannot use File Image as the fit model with Wasm build.
Please use local build if you want that feature (it is faster to train on local build anyway).
