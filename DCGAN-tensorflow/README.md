# DCGAN in Tensorflow

Tensorflow implementation of [Deep Convolutional Generative Adversarial Networks](http://arxiv.org/abs/1511.06434) The referenced torch code can be found [here](https://github.com/carpedm20/DCGAN-tensorflow).

![alt tag](DCGAN.png)


## Prerequisites

- Python 3
- [Tensorflow 1.5]
- [SciPy](http://www.scipy.org/install.html)
- [pillow](https://github.com/python-pillow/Pillow)


## Usage

First, download dataset CelabA and Standford Cars.

To train a model with prepocessed dataset:

    $ python main.py --dataset celebA --input_height=64 --train 
    $ python main.py --dataset standford_cars --input_height=64 --train 

## Results

![result](assets/faces.png)

## Author

Wentian Bao wb2328
Zhang Zhang zz2517
Haotian Zhang hz2475
