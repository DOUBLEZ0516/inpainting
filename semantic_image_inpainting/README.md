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

We also made an improvement on trianing weight mask W.

Prerequisit
------------
 - Tensorflow = 1.7.0
 - scipy + PIL/pillow (image io)
 - pyamg (for Poisson blending)

Tested to work with Python 3.5


Files
-----
 - src/model.py: implementation of semantic inpaint model
 - src/inpaint.py: command line application, which passes user defined parameters to model.py
 - src/external: external code used. Includs Poisson blending.
 - graphs：contains pre-trained gan .pb files
     - dcgan-100.pb: gan trained on 64 * 64 CelebA
     - model2.pb: gan trained on 64 * 64 Stanford Cars
     - model_128_64_64_1.pb: gan trained on 128 * 128 Stanford Car

Running
-------
To generate result on 64 * 64 face data set
```
python3 src/inpaint.py --model_file graphs/dcgan-100.pb \
    --maskType center --inDir testimages \
    --nIter 1000 --blend --Wstep 100
```

To generate result on 64 * 64 car data set
```
python3 src/inpaint.py --model_file graphs/model2.pb \
    --maskType random --inDir testcars \
    --nIter 1000 --blend --Wstep 100 --imgExt jpg
```

To generate result on improved resolution of resized images of cars \
for images resized from 64 * 64 to 128 * 128
```
python3 src/inpaint.py --model_file graphs/model_128_64_64_1.pb \
    --maskType mask_64_128 --inDir testcars_128 --imgSize 128\
    --nIter 1000 --blend --Wstep 1000 --imgExt jpeg
```

for images resized from 32 * 32 to 128 * 128
```
python3 src/inpaint.py --model_file graphs/model_128_64_64_1.pb \
    --maskType mask_32_128 --inDir testcars_128 --imgSize 128\
    --nIter 1000 --blend --Wstep 1000 --imgExt jpeg
```
The inpainted images will be stored at file ./completions. Both generated images that are used to inpaint and inpainted images will be stored.
