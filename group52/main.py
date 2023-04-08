import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch as th
import torchvision.models.detection
from PIL import Image
from torchvision.models.detection import RetinaNet_ResNet50_FPN_V2_Weights

from group52.gen_isp import load_annotations, GenISP, annotations_to_tensor
from group52.image_helper import auto_post_process_image, load_image
from group52.retinanet_helper import create_label_dictionary


def get_hyper_parameters():
    return {
        'epochs': 200,
        'batch_size': 2,

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
        'resize': (1333, 800),
        'class_list': '../data/class_list.csv',

    }


def main(preprocess_images=False):
    h_parameters = get_hyper_parameters()

    gen_isp = GenISP()

    labels = create_label_dictionary(h_parameters['class_list'])
    print('Loaded labels:', labels)

    # Load the model https://github.com/pytorch/vision/blob/main/torchvision/models/detection/retinanet.py
    object_detector = torchvision.models.detection.retinanet_resnet50_fpn_v2(weights=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1)
    object_detector.requires_grad_(False)
    th.save(object_detector, '../data/retinanet_v2_object_detector.pickle')

    data_dir = '../data/our_sony/'
    # load_label
    targets_per_image = load_annotations(data_dir + 'raw_new_train.json')
    targets_per_image = load_annotations(data_dir + 'raw_new_test.json', targets_per_image)
    targets_per_image = load_annotations(data_dir + 'raw_new_val.json', targets_per_image)
    targets_per_image = annotations_to_tensor(targets_per_image)

    raw_images_dir = data_dir + 'raw_images/'
    processed_images_dir = data_dir + 'processed_images/'
    gen_isp_images_dir = data_dir + 'gen_isp_images/'

    if preprocess_images:
        images_paths = os.listdir(raw_images_dir)
        for p in images_paths:
            image_id = p.split('.')[0]
            # image = (load_image(raw_images_dir + p)*255).astype(np.uint8)
            image = auto_post_process_image(raw_images_dir + p)
            Image.fromarray(image).resize(h_parameters['resize']).save(processed_images_dir + image_id + '.png', format='png')
            print(f'Saved image to: {processed_images_dir + image_id + ".png"}')

    images_paths = os.listdir(processed_images_dir)
    for epoch in range(h_parameters['epochs']):
        epoch_loss = []
        batch_inputs, batch_targets = [], []
        for i, p in enumerate(images_paths):
            print('Training on image:', p)
            image_id = p.split('.')[0]
            targets = targets_per_image[image_id]
            image_np_array = cv2.imread(os.path.join(processed_images_dir, p))
            image_tensor = th.from_numpy(image_np_array).unsqueeze(0).permute(0, 3, 1, 2).div(255.0)
            batch_inputs.append(image_tensor)
            batch_targets.append(targets)

            # If we have a full batch, train the model
            if len(batch_inputs) % h_parameters['batch_size'] == 0:
                gen_isp_outputs = gen_isp(batch_inputs)
                object_detector_losses = object_detector(gen_isp_outputs, batch_targets)
                gen_isp.optimizer.zero_grad()
                total_loss = object_detector_losses['classification'] + object_detector_losses['bbox_regression']
                total_loss.backward()
                gen_isp.optimizer.step()
                epoch_loss.append(total_loss.item())
                batch_inputs, batch_targets = [], []
        print(f'{epoch} | Epoch loss: {np.mean(epoch_loss)}+/-{np.std(epoch_loss)}')
        for p in images_paths:
            image_id = f'{epoch}_{p.split(".")[0]}'
            image_np_array = cv2.imread(os.path.join(processed_images_dir, p))
            image_tensor = th.from_numpy(image_np_array).unsqueeze(0).permute(0, 3, 1, 2).div(255.0)
            with th.no_grad():
                gen_isp_outputs = gen_isp([image_tensor])
            gen_isp_array = (gen_isp_outputs[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            Image.fromarray(gen_isp_array).save(gen_isp_images_dir + image_id + '.png', format='png')
            print(f'Saved image to: {gen_isp_images_dir + image_id + ".png"}')


if __name__ == '__main__':
    main()
