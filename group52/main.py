from copy import deepcopy

import torch as th
import os
import rawpy
import numpy as np

from group52.retinanet import model


class GenISP(th.nn.Module):

    def __init__(self):
        super().__init__()
        # minimal pre-processing pipeline packing and color space transformation

        # 2-step color processing stage realized by image-to-parameter modules: ConvWB and ConvCC
        self.image_to_parameter = th.nn.Sequential(
            th.nn.Conv2d(3, 16, 7), th.nn.LeakyReLU(), th.nn.MaxPool2d(),
            th.nn.Conv2d(16, 32, 5), th.nn.LeakyReLU(), th.nn.MaxPool2d(),
            th.nn.Conv2d(32, 128, 3), th.nn.LeakyReLU(), th.nn.MaxPool2d(),
            th.nn.AdaptiveAvgPool2d(1)
        )
        self.conv_wb = th.nn.Sequential(deepcopy(self.image_to_parameter),
                                        th.nn.Linear(3, 1))
        self.conv_cc = th.nn.Sequential(deepcopy(self.image_to_parameter),
                                        th.nn.Linear(9, 1))

        # A non-linear local image enhancement by a shallow ConvNet
        self.shallow_conv_net = th.nn.Sequential(th.nn.Conv2d(3, 16, 3), th.nn.InstanceNorm2d(), th.nn.LeakyReLU(),
                                                 th.nn.Conv2d(16, 64, 3), th.nn.InstanceNorm2d(), th.nn.LeakyReLU(),
                                                 th.nn.Conv2d(64, 3, 1))

# https://www.quora.com/What-is-the-RGB-to-XYZ-conversion-matrix-Is-it-possible-to-convert-from-RGB-to-XYZ-using-just-this-matrix-no-other-information-If-not-why-not
def load_image(path):
    with rawpy.imread(path) as raw:
        conversion_matrix = raw.rgb_xyz_matrix
        raw_img = raw.raw_image
        h, w = raw.shape[0], raw.shape[1]
        xyz_img = np.ndarray(raw.shape)

        # average green channels form the packed rerpesentation

        #apply CST matrix
        for h_i in range(h):
            for w_i in range(w):
                xyz_img[h_i, w_i] = np.matmul(conversion_matrix, raw_img[h_i, w_i])

    return xyz_img


    


def main():
    object_detector = model.resnet50(num_classes=80)
    object_detector.load_state_dict(th.load('../data/coco_resnet_50_map_0_335_state_dict.pt', map_location=th.device('cpu')))
 
    images_paths = os.listdir('../data/sony_raw/')

    for p in images_paths:
        image = load_image(p)


if __name__ == '__main__':
    main()
