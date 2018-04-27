Semantic Image Inpainting With Deep Generative Models
=====================================================
Implementation of Semantic Image Inpainting With Deep Generative Models

[Raymond A. Yeh*](http://www.isle.illinois.edu/~yeh17/),
[Chen Chen*](http://cchen156.web.engr.illinois.edu/),
[Teck Yian Lim](http://tlim11.web.engr.illinois.edu/),
[Alexander G. Schwing](http://www.alexander-schwing.de/),
[Mark Hasegawa-Johnson](http://www.ifp.illinois.edu/~hasegawa/),
[Minh N. Do](http://minhdo.ece.illinois.edu/)

In CVPR 2017


Overview
--------
Implementation of proposed cost function and backpropogation to input. 

In this code release, we load a pretrained DCGAN model, and apply our proposed
objective function for the task of image completion

Dependencies
------------
 - Tensorflow >= 1.0
 - scipy + PIL/pillow (image io)
 - pyamg (for Poisson blending)

Tested to work with both Python 2.7 and Python 3.5


Files
-----
 - src/model.py - implementation of semantic inpaint model
 - src/inpaint.py - command line application, which passes user defined parameters to model.py
 - src/external - external code used. Includs Poisson blending.
 - graphs - contains pre-trained gan .pb files

Running
-------
Train and generate for test images
```
python src/inpaint.py --model_file graphs/dcgan-100.pb \
    --maskType center --inDir testimages \
    --nIter 1000 --blend
```


