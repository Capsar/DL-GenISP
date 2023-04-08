import json

import numpy as np
import torch as th
import torch.nn.functional as F
from copy import deepcopy


class Diagonalize(th.nn.Module):

    def __init__(self):
        super().__init__()
        self.diag = th.diag

    def forward(self, x):
        return self.diag(x.squeeze())


class Resize(th.nn.Module):

    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, x):
        return F.interpolate(x, size=self.size, mode='bilinear')


class GenISP(th.nn.Module):

    def __init__(self):
        super().__init__()
        # minimal pre-processing pipeline packing and color space transformation

        # 2-step color processing stage realized by image-to-parameter modules: ConvWB and ConvCC
        self.image_to_parameter = th.nn.Sequential(
            th.nn.Conv2d(3, 16, kernel_size=7, padding=3), th.nn.LeakyReLU(), th.nn.MaxPool2d(kernel_size=2),
            th.nn.Conv2d(16, 32, kernel_size=5, padding=2), th.nn.LeakyReLU(), th.nn.MaxPool2d(kernel_size=2),
            th.nn.Conv2d(32, 128, kernel_size=3, padding=1), th.nn.LeakyReLU(), th.nn.MaxPool2d(kernel_size=2),
            th.nn.AdaptiveAvgPool2d(1),
            th.nn.Flatten(1),
        )

        self.conv_wb = th.nn.Sequential(
            Resize((256, 256)),
            deepcopy(self.image_to_parameter),
            th.nn.Linear(128, 3),
            Diagonalize(),
        )

        self.conv_cc = th.nn.Sequential(
            Resize((256, 256)),
            deepcopy(self.image_to_parameter),
            th.nn.Linear(128, 9),
            th.nn.Unflatten(1, (3, 3))
        )

        # A non-linear local image enhancement by a shallow ConvNet
        self.shallow_conv_net = th.nn.Sequential(th.nn.Conv2d(3, 16, kernel_size=3, padding=1), th.nn.InstanceNorm2d(16), th.nn.LeakyReLU(),
                                                 th.nn.Conv2d(16, 64, kernel_size=3, padding=1), th.nn.InstanceNorm2d(64), th.nn.LeakyReLU(),
                                                 th.nn.Conv2d(64, 3, kernel_size=1))

        self.optimizer = th.optim.Adam(self.parameters(), lr=1e-2)

    def forward(self, batch):
        """
        :param batch: batch of images
        :return: enhanced images
        """
        output = []
        for image in batch:
            wb_matrix = self.conv_wb(image)
            image = th.matmul(image.permute(0, 2, 3, 1), wb_matrix).permute(0, 3, 1, 2)
            cc_matrix = self.conv_cc(image)
            image = th.matmul(image.permute(0, 2, 3, 1), cc_matrix).permute(0, 3, 1, 2)
            x = self.shallow_conv_net(image)
            output.append(x.squeeze(0))
        return output


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


def load_annotations(file_path, previous_dictionary=None):
    with open(file_path, 'r') as f:
        data = json.load(f)
        annotation_dict = {}
        if previous_dictionary is not None:
            annotation_dict = previous_dictionary

        for image in data['annotations']:
            if image['image_id'] in annotation_dict.keys():
                bbox = [image['bbox'][0], image['bbox'][1], image['bbox'][0] + image['bbox'][2], image['bbox'][1] + image['bbox'][3]]
                annotation_dict[image['image_id']]['boxes'].append(bbox)
                annotation_dict[image['image_id']]['labels'].append(image['category_id'])
            else:
                annotation_dict[image['image_id']] = {'boxes': [], 'labels': []}
        return annotation_dict


def annotations_to_tensor(annotation_dict):
    for image_id in annotation_dict.keys():
        annotation_dict[image_id]['boxes'] = th.tensor(annotation_dict[image_id]['boxes'])
        annotation_dict[image_id]['labels'] = th.tensor(annotation_dict[image_id]['labels'])
    return annotation_dict
