from __future__ import division
from configparser import ConfigParser
import argparse
import os
from model import AELikeModel


def main(args):
    # parser config
    cp = ConfigParser()
    cp.read(args.config)
    model_path= args.model
    input_image = args.input
    output_image = args.output
    image_size = cp["TRAIN"].getint("image_size")
    alpha = cp["TRAIN"].getfloat("alpha")
    model = AELikeModel(image_size, alpha,AE = True, verbose = False, trained_model=model_path)
    print(os.path.abspath(input_image))
    model.suppress(input_image, output_image)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bone Suppression Testing')
    parser.add_argument('--model', default='model/model', type=str, help='model path')
    parser.add_argument('--config', default='config/train.cfg', type=str, help='model config')
    parser.add_argument('--input', default='test.png', type=str, help='input image')
    parser.add_argument('--output', default='output.png', type=str, help='output image')
    args = parser.parse_args()
    main(args)
