from copy import deepcopy

import torch as th
import os
import rawpy
import numpy as np

from group52.retinanet import model
from matplotlib import pyplot as plt

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


def load_image(path):
    """
        Source: https://www.quora.com/What-is-the-RGB-to-XYZ-conversion-matrix-Is-it-possible-to-convert-from-RGB-to-XYZ-using-just-this-matrix-no-other-information-If-not-why-not
    :param path: path to the raw image.
    :return: loaded image.
    """
    print(path)
    with rawpy.imread(path) as raw:
        print(raw)

        conversion_matrix = raw.rgb_xyz_matrix
        raw_image = raw.raw_image_visible.astype(np.int32)
        new_image = np.zeros((int(raw_image.shape[0]/2), int(raw_image.shape[1]/2), 4), dtype=np.int32)
        new_image[:, :, 0] = raw_image[0::2, 0::2]
        new_image[:, :, 1] = raw_image[0::2, 1::2]
        new_image[:, :, 2] = raw_image[1::2, 0::2]
        new_image[:, :, 3] = raw_image[1::2, 1::2]

        print(new_image.shape)
        plt.imshow(new_image[:, :, 2])
        plt.show()

        # average green channels form the packed representation

        # # apply CST matrix
        # for h_i in range(h):
        #     for w_i in range(w):
        #         print(conversion_matrix.shape, raw_img[h_i, w_i].shape)
        #         xyz_img[h_i, w_i] = np.matmul(conversion_matrix, raw_img[h_i, w_i])

    return packed


def main():
    object_detector = model.resnet50(num_classes=80)
    object_detector.load_state_dict(th.load('../data/coco_resnet_50_map_0_335_state_dict.pt', map_location=th.device('cpu')))

    data_dir = '../data/our_sony/'
    images_paths = os.listdir(data_dir)
    for p in images_paths:
        image = load_image(data_dir + p)
        break


if __name__ == '__main__':
    main()
