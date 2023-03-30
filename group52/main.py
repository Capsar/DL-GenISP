import math
from copy import deepcopy

import torch as th
import torch.nn.functional as F
import os
import rawpy
import numpy as np

from group52.retinanet import model
from matplotlib import pyplot as plt


def load_image(path):
    """
        Source: https://www.quora.com/What-is-the-RGB-to-XYZ-conversion-matrix-Is-it-possible-to-convert-from-RGB-to-XYZ-using-just-this-matrix-no-other-information-If-not-why-not
    :param path: path to the raw image.
    :return: loaded image.
    """
    print(path)
    with rawpy.imread(path) as raw:
        print(raw)

        # pack
        conversion_matrix = raw.rgb_xyz_matrix
        raw_image = raw.raw_image_visible.astype(np.int32)
        packed_image = np.zeros((int(raw_image.shape[0] / 2), int(raw_image.shape[1] / 2), 4), dtype=np.int32)
        packed_image[:, :, 0] = raw_image[0::2, 0::2]
        packed_image[:, :, 1] = raw_image[0::2, 1::2]
        packed_image[:, :, 2] = raw_image[1::2, 0::2]
        packed_image[:, :, 3] = raw_image[1::2, 1::2]
        print(packed_image.shape)

        # averaged green channel
        averaged_image = np.zeros((int(raw_image.shape[0] / 2), int(raw_image.shape[1] / 2), 3), dtype=np.int32)
        averaged_image[:, :, 0] = packed_image[:, :, 0]
        averaged_image[:, :, 1] = packed_image[:, :, 1]
        averaged_image[:, :, 2] = (packed_image[:, :, 2] + packed_image[:, :, 3]) / 2
        print(averaged_image.shape)
        averaged_max_value = np.max(averaged_image)
        normalized_averaged_image = averaged_image / averaged_max_value
        print(normalized_averaged_image.shape, normalized_averaged_image[0, 0])
        plt.imshow(normalized_averaged_image)
        plt.show()

        # convert color channel
        conversion_matrix = raw.rgb_xyz_matrix
        print(conversion_matrix, conversion_matrix.shape)

        xyz_image = averaged_image @ conversion_matrix.T
        print(xyz_image.shape)
        max_value = np.max(xyz_image)

        normalized_image = xyz_image / max_value
        print(normalized_image.shape, normalized_image[0, 0, 0:3])

        plt.imshow(normalized_image[:, :, 0:3])
        plt.show()
    return xyz_image


class Diagonalize(th.nn.Module):

    def __init__(self):
        super().__init__()
        self.diag = th.diag

    def forward(self, x):
        return self.diag(x)


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

        self.conv_wb = th.nn.Sequential(
            F.interpolate(size=(256, 256), mode='bilinear'),
            deepcopy(self.image_to_parameter),
            th.nn.Linear(128, 3),
            Diagonalize(),
        )

        self.conv_cc = th.nn.Sequential(
            F.interpolate(size=(256, 256), mode='bilinear'),
            deepcopy(self.image_to_parameter),
            th.nn.Linear(128, 9),
            th.nn.Unflatten(1, (3, 3))
        )

        # A non-linear local image enhancement by a shallow ConvNet
        self.shallow_conv_net = th.nn.Sequential(th.nn.Conv2d(3, 16, 3), th.nn.InstanceNorm2d(), th.nn.LeakyReLU(),
                                                 th.nn.Conv2d(16, 64, 3), th.nn.InstanceNorm2d(), th.nn.LeakyReLU(),
                                                 th.nn.Conv2d(64, 3, 1))

    def forward(self, batch):
        """
        :param batch: batch of images
        :return: enhanced images
        """
        x = self.conv_wb(batch)
        x = self.conv_cc(x)
        x = self.shallow_conv_net(x)
        return x


def regression_loss(object_detector):
    """
    What do we do here?
    Paper says 'regression loss [is implemented by] by smooth-L1 loss'
    """
    # TODO! let's see if it works :)) (finger crossed)
    loss = torch.nn.SmoothL1Loss()

    retinanet = model.resnet50(num_classes=dataset_train.num_classes(),)
    retinanet.load_state_dict(torch.load(PATH_TO_WEIGHTS))
    
    return loss(object_detector, retinanet(original_image))


def penalize_intensive_colors(enhanced_image):
    """
    However, we observe than when trained with an object
    detection dataset without any constraints, GenISP reduces
    color saturation and loses colors not important for
    discrimination of objects in the dataset. If keeping the colors is
    important, an additional loss term can be employed:
    See equation (4) in the paper
    This loss term inspired by gray-world hypothesis encourages the model to
    balance the average intensity between each color channel.
    """

    avg_intensity = np.mean(enhanced_image, axis=(0, 1))
    loss = abs(avg_intensity[0] - avg_intensity[1]) + abs(avg_intensity[0] - avg_intensity[2]) + abs(avg_intensity[1] - avg_intensity[2])
    return loss

def calculate_loss(object_detector: model, keep_colors: bool = False):  # I am quite sure we need some other parameters
    """
    The total loss is composed of classification and regression loss:
    L_total = L_cls + L_reg

    As for our experiments, we use RetinaNet as the detector
    for guiding GenISP during the training, so the classification
    loss is implemented by α-balanced focal loss [16] and regression
    loss by smooth-L1 loss
    """
    loss = classification_loss(object_detector) + regression_loss(object_detector)
    if keep_colors:
        loss += weight * penalize_intensive_colors()
    return loss


def classification_loss(object_detector):
    # In RetinaNet classification loss is implemented by α-balanced focal loss
    # TODO! Check if correct
    return object_detector.focalLoss


def get_hyper_parameters():
    return {
        'epochs': 12,
        'batch_size': 8,

        # Key states from which epoch the learning rate is enabled,
        # Value is the learning rate.
        #
        # We set the learning rate to 1e−2 initially and
        # decrease it to 1e−3 and 1e−4 at the 5th and
        # 10th epoch, respectively.
        'learning_rates': {
            0, math.exp(-2),
            5, math.exp(-3),
            10, math.exp(-4)
        },
        # During training and testing, we
        # resize the images to a maximum size of 1333 × 800 and
        # keep the image aspect ratio. In ConvWB and ConvCC, we
        # resize input to 256 × 256 using bilinear interpolation
        'resized_image_height': 800,

    }


def main():
    # object_detector = model.resnet50(num_classes=80)
    # object_detector.load_state_dict(th.load('../data/coco_resnet_50_map_0_335_state_dict.pt', map_location=th.device('cpu')))

    data_dir = '../data/our_sony/'
    images_paths = os.listdir(data_dir)
    for p in images_paths:
        image = load_image(data_dir + p)
        break


if __name__ == '__main__':
    print(get_hyper_parameters())
    main()
