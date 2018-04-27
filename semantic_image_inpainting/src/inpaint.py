import tensorflow as tf
import scipy.misc
import argparse
import os
import numpy as np
from glob import glob

from model import ModelInpaint

parser = argparse.ArgumentParser()
parser.add_argument('--model_file', type=str, help="Pretrained GAN model")
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--nIter', type=int, default=1000)
parser.add_argument('--imgSize', type=int, default=64)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lambda_p', type=float, default=0.1)
parser.add_argument('--checkpointDir', type=str, default='checkpoint')
parser.add_argument('--outDir', type=str, default='completions')
parser.add_argument('--blend', action='store_true', default=False,
                    help="Blend predicted image to original image")
parser.add_argument('--maskType', type=str,
                    choices=['random', 'center', 'left', 'mask_64_128', 'mask_32_128'],
                    default='center')

parser.add_argument('--maskThresh', type=int,
                    default=128,
                    help='Threshold in case input mask is not binary')
parser.add_argument('--in_image', type=str, default=None,
                    help='Input Image (ignored if inDir is specified')
parser.add_argument('--inDir', type=str, default=None,
                    help='Path to input images')
parser.add_argument('--imgExt', type=str, default='png',
                    help='input images file extension')

args = parser.parse_args()


def loadimage(filename):
  """
  This function loads input images
  """
    img = scipy.misc.imread(filename, mode='RGB').astype(np.float)
    return img


def saveimages(outimages, prefix='samples'):
  """
  This function writes generated images
  """
    numimages = len(outimages)

    if not os.path.exists(args.outDir):
        os.mkdir(args.outDir)

    for i in range(numimages):
        filename = '{}_{}.png'.format(prefix, i)
        filename = os.path.join(args.outDir, filename)
        scipy.misc.imsave(filename, outimages[i, :, :, :])


def gen_mask(maskType):
  """
  This function generates input masks
  """
  
    image_shape = [args.imgSize, args.imgSize]
    if maskType == 'random':
        fraction_masked = 0.2
        mask = np.ones(image_shape)
        mask[np.random.random(image_shape[:2]) < fraction_masked] = 0.0
    elif maskType == 'center':
        scale = 0.25
        assert(scale <= 0.5)
        mask = np.ones(image_shape)
        sz = args.imgSize
        l = int(args.imgSize*scale)
        u = int(args.imgSize*(1.0-scale))
        mask[l:u, l:u] = 0.0
    elif maskType == 'left':
        mask = np.ones(image_shape)
        c = args.imgSize // 2
        mask[:, :c] = 0.0
    elif maskType == 'mask_64_128':
        mask = np.ones([128,128])
        for i in range(64):
            mask[2*i,:] = 0
            mask[:,2*i] = 0
    elif maskType == 'mask_32_128':
        mask = np.zeros([128, 128])
        mask[:,::4][::4,:] = 1
    else:
        assert(False)
    return mask


def main():
  """
  Main function
  """
    m = ModelInpaint(args.model_file, args)

    # Generate some samples from the model as a test
    imout = m.sample()
    saveimages(imout)

    mask = gen_mask(args.maskType)
    if args.inDir is not None:
        imgfilenames = glob( args.inDir + '/*.' + args.imgExt )
        print('{} images found'.format(len(imgfilenames)))
        in_img = np.array([loadimage(f) for f in imgfilenames])
    elif args.in_image is not None:
        in_img = loadimage(args.in_image)
    else:
        print('Input image needs to be specified')
        exit(1)

    inpaint_out, g_out = m.inpaint(in_img, mask, args.blend)
    scipy.misc.imsave(os.path.join(args.outDir, 'mask.png'), mask)
    saveimages(g_out, 'gen')
    saveimages(inpaint_out, 'inpaint')


if __name__ == '__main__':
    main()
