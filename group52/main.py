import csv
import json
import math
import os

import torch as th
import torchvision.models.detection
from torchvision.models.detection import RetinaNet_ResNet50_FPN_V2_Weights

from group52 import visualize_single_image, retinanet_helper
from group52.gen_isp import load_annotations, GenISP
from group52.image_helper import auto_post_process_image
from group52.retinanet_helper import create_label_dictionary, process_model_output

# Load focal loss model from https://github.com/AdeelH/pytorch-multi-class-focal-loss
focal_loss = torch.hub.load(
    'adeelh/pytorch-multi-class-focal-loss',
    model='focal_loss',
    alpha=[.75, .25],
    gamma=2,
    reduction='mean',
    device='cpu',
    dtype=torch.float32,
    force_reload=False
)

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
            0, 1e-2,
            5, 1e-3,
            10, 1e-4
        },
        # During training and testing, we
        # resize the images to a maximum size of 1333 × 800 and
        # keep the image aspect ratio. In ConvWB and ConvCC, we
        # resize input to 256 × 256 using bilinear interpolation
        'resized_image_height': 800,
        'detection_threshold': 0.6,
        'class_list': '../data/class_list.csv',

    }



def main():
    h_parameters = get_hyper_parameters()

    gen_isp = GenISP()

    labels = create_label_dictionary(h_parameters['class_list'])

    object_detector = torchvision.models.detection.retinanet_resnet50_fpn_v2(weights=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1)
    th.save(object_detector, '../data/retinanet_v2_object_detector.pickle')
    object_detector.eval()

    data_dir = '../data/our_sony/'
    # load_label
    annotations_dict = load_annotations(data_dir + 'raw_new_train.json')
    annotations_dict = load_annotations(data_dir + 'raw_new_test.json', annotations_dict)
    annotations_dict = load_annotations(data_dir + 'raw_new_val.json', annotations_dict)

    raw_images_dir = data_dir + 'raw_images/'
    images_paths = os.listdir(raw_images_dir)
    for p in images_paths:
        image_id = p.split('.')[0]
        annotations = annotations_dict[image_id]

        image_np_array = auto_post_process_image(raw_images_dir + p)
        image_tensor = th.from_numpy(image_np_array).unsqueeze(0).permute(0, 3, 1, 2).div(255.0)
        enhanced_image = gen_isp(image_tensor)
        with th.no_grad():
            outputs = object_detector(enhanced_image)
            output_boxes, output_categories, output_classes = process_model_output(outputs, h_parameters['detection_threshold'], labels)

        print(annotations)
        print(list(zip(output_boxes, output_categories)))
        gen_isp.optimizer.zero_grad()
        # TODO: Calculate loss between annotations and predictions (output_boxes, output_categories)
        classification_loss = classification_loss(output_categories, annotations)
        regression_loss = regression_loss(output_categories, annotations)
        total_loss = classification_loss + regression_loss

        # Backpropagate loss & take optimizer step
        total_loss.backward()
        gen_isp.optimizer.step()

# Calculate focal loss as classification loss
def classification_loss(output_categories, annotations):
    loss = focal_loss(output_categories, annotations)

# Calculate smooth L1 loss as regression loss
def regression_loss(output_categories, annotations):
    smoothl1loss = torch.nn.SmoothL1Loss()
    return smoothl1loss(output_categories, annotations)

if __name__ == '__main__':
    main()
