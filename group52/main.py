from copy import deepcopy

import torch as th

from group52.retinanet import model


class GenISP(th.nn.Module):

    def __init__(self):
        super().__init__()
        # minimal pre-processing pipeline packing and color space transformation

        # 2-step color processing stage realized by image-to-parameter modules: ConvWB and ConvCC
        self.image_to_parameter = th.nn.Sequential(
            th.nn.Conv2d(3, 16, 7, 1), th.nn.LeakyReLU(), th.nn.MaxPool2d(),
            th.nn.Conv2d(16, 32, 5, 1), th.nn.LeakyReLU(), th.nn.MaxPool2d(),
            th.nn.Conv2d(32, 128, 3, 1), th.nn.LeakyReLU(), th.nn.MaxPool2d(),
            th.nn.AdaptiveAvgPool2d(1)
        )
        self.conv_wb = th.nn.Sequential(deepcopy(self.image_to_parameter),
                                        th.nn.Linear(3, 1))
        self.conv_cc = th.nn.Sequential(deepcopy(self.image_to_parameter),
                                        th.nn.Linear(9, 1))

        # A non-linear local image enhancement by a shallow ConvNet
        self.shallow_conv_net = th.nn.Sequential(th.nn.Conv2d(3, 16, 3, 1), th.nn.InstanceNorm2d(), th.nn.LeakyReLU(),
                                                 th.nn.Conv2d(16, 64, 3, 1), th.nn.InstanceNorm2d(), th.nn.LeakyReLU(),
                                                 th.nn.Conv2d(64, 3, 1, 1))


def main():
    object_detector = model.resnet50(num_classes=80)
    object_detector.load_state_dict(th.load('../data/coco_resnet_50_map_0_335_state_dict.pt', map_location=th.device('cpu')))


if __name__ == '__main__':
    main()
