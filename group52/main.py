import math
from copy import deepcopy

import torch as th

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


def regression_loss(object_detector):
    """
    What do we do here?
    Paper says 'regression loss [is implemented by] by smooth-L1 loss'
    """
    # TODO! implement
    return 0


def penalize_intensive_colors():
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
    # TODO! implement
    return 0



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
        loss += penalize_intensive_colors()
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
    object_detector = model.resnet50(num_classes=80)
    object_detector.load_state_dict(
        th.load('../data/coco_resnet_50_map_0_335_state_dict.pt', map_location=th.device('cpu')))


if __name__ == '__main__':
    print(get_hyper_parameters())
    main()
