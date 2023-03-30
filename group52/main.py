import math
from copy import deepcopy

import torch as th
import torch.nn.functional as F
import os
import rawpy
import numpy as np
from rawpy._rawpy import ColorSpace

from group52.retinanet import model
from matplotlib import pyplot as plt
import json


def load_image(path):
    """
        Source: https://www.quora.com/What-is-the-RGB-to-XYZ-conversion-matrix-Is-it-possible-to-convert-from-RGB-to-XYZ-using-just-this-matrix-no-other-information-If-not-why-not
        :param path: path to the raw image.
        :return: loaded image.
    """
    print('Loading image: {}'.format(path))
    with rawpy.imread(path) as raw:

        # pack
        raw_image = raw.raw_image.astype(np.int32)
        packed_image = np.zeros((int(raw_image.shape[0] / 2), int(raw_image.shape[1] / 2), 4), dtype=np.int32)
        packed_image[:, :, 0] = raw_image[0::2, 0::2] # R Left top
        packed_image[:, :, 1] = raw_image[0::2, 1::2] # G Right top
        packed_image[:, :, 2] = raw_image[1::2, 0::2] # G Left bottom
        packed_image[:, :, 3] = raw_image[1::2, 1::2] # B Right bottom

        # averaged green channel
        averaged_image = np.zeros((packed_image.shape[0], packed_image.shape[1], 3), dtype=np.int32)
        averaged_image[:, :, 0] = packed_image[:, :, 0] # R
        averaged_image[:, :, 1] = (packed_image[:, :, 1] + packed_image[:, :, 2]) # G
        averaged_image[:, :, 2] = packed_image[:, :, 3] # B

        # convert color channel
        conversion_matrix = raw.rgb_xyz_matrix

        # Or from packed or from averaged
        xyz_image = packed_image @ conversion_matrix
        plt.imshow(xyz_image / 2**13)
        plt.show()

        xyz_image = averaged_image @ conversion_matrix[0:3, 0:3]
        plt.imshow(xyz_image / 2**13)
        plt.show()

        return th.from_numpy(xyz_image).float()


def auto_post_process_image(path):
    print('Loading image: {}'.format(path))
    with rawpy.imread(path) as raw:
        post_processed_image = raw.postprocess(half_size=True, no_auto_bright=True, output_color=ColorSpace.XYZ)
        plt.imshow(post_processed_image)
        plt.show()
        print(post_processed_image.shape)

        return th.from_numpy(post_processed_image).float()

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

    # gen_isp = GenISP()
    object_detector = model.resnet50(num_classes=80, pretrained=True, model_dir='../data/')
    th.save(object_detector, '../data/resnet50_object_detector.pickle')
    object_detector.eval()

    data_dir = '../data/our_sony/'
    # load_labels
    f = open(data_dir + 'raw_new_train.json')
    data = json.load(f)
    data_dict = {}
    for image in data['annotations']:
        if image['image_id'] in data_dict.keys():
            data_dict[image['image_id']].append([image['bbox'], image['category_id']])
        else:
            data_dict[image['image_id']] = [[image['bbox'], image['category_id']]]

    print(data.keys())

    raw_images_dir = data_dir + 'raw_images/'
    images_paths = os.listdir(raw_images_dir)
    for p in images_paths:
        image_id = p.split('.')[0]
        annotations = data_dict[image_id]
        image = load_image(raw_images_dir + p).unsqueeze(0)
        image = image.reshape(1, image.shape[-1], image.shape[1], image.shape[2])
        #
        # print(image.shape, annotations)
        # # enhanced_image = gen_isp(image)
        y_pred = object_detector(image)
        print(y_pred)
        # reg_loss = regression_loss(y_pred, y_true)




if __name__ == '__main__':
    print(get_hyper_parameters())
    main()
