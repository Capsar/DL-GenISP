from copy import deepcopy

import torch as th
import os
import rawpy
import numpy as np

# from group52.retinanet import model


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
        
        raw_image = raw.raw_image
        h, w = raw_image.shape[0], raw_image.shape[1] 
        print(raw_image.shape)

        


        for i in range(h):
            for j in range(w):
                p1 = raw_image[i,j]
                p2 = raw_image[i+1, j]
                p3 = raw_image[i, j+1]
                p4 = raw_image[i+1, j+1]


            
        conversion_matrix = raw.rgb_xyz_matrix
        h, w = raw.raw_image.shape[0], raw.raw_image.shape[1]


        xyz_img = raw.postprocess(four_color_rgb=True, output_color=rawpy.ColorSpace(XYZ))
        # pack image



    return xyz_img


    


def main():
    # object_detector = model.resnet50(num_classes=80)
    # object_detector.load_state_dict(th.load('../data/coco_resnet_50_map_0_335_state_dict.pt', map_location=th.device('cpu')))
 
    # images_paths = os.listdir('../data/sony_raw/')

    im = load_image('/Users/taichi/Downloads/our_nikon_and_sony/our_sony/DSC01088.ARW')
    print(load_image)


if __name__ == '__main__':
    main()
