import os
import random

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
    """
    Hyperparameters from the GenISP paper .
    For reproducibility, we set the random seed to 42 for all experiments.
    :return: dictionary with hyperparameters.
    """

    seed = 42
    th.random.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    print(f'Running experiment with seed: {seed}')
    return {
        'epochs': 200,
        'batch_size': 2,
        'resize': (1333, 800), # During training and testing, we resize the images to a maximum size of 1333 × 800 and keep the image aspect ratio.
        'class_list': '../data/class_list.csv',

    }


def main(postprocess_images=False):
    h_parameters = get_hyper_parameters()

    gen_isp = GenISP()

    # Load the model https://github.com/pytorch/vision/blob/main/torchvision/models/detection/retinanet.py
    object_detector = torchvision.models.detection.retinanet_resnet50_fpn_v2(weights=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1)
    # We do not want to train the object detector so set requires_grad to false for all parameters.
    for param in object_detector.parameters():
        param.requires_grad = False
    th.save(object_detector, '../data/retinanet_v2_object_detector.pickle')

    data_dir = '../data/our_sony/'
    # load_label
    targets_per_image = load_annotations(data_dir + 'raw_new_train.json')
    targets_per_image = load_annotations(data_dir + 'raw_new_test.json', targets_per_image)
    targets_per_image = load_annotations(data_dir + 'raw_new_val.json', targets_per_image)
    targets_per_image = annotations_to_tensor(targets_per_image)

    raw_images_dir = data_dir + 'raw_images/' # Containing the .ARW files.
    processed_images_dir = data_dir + 'processed_images/' # Containing the .PNG files outputted by RawPy.
    gen_isp_images_dir = data_dir + 'gen_isp_images/' # Containing the .PNG files outputted by GenISP. (Intermediate results)

    # Post-process the images using RawPy post-processing.
    if postprocess_images:
        images_paths = os.listdir(raw_images_dir)
        for p in images_paths:
            image_id = p.split('.')[0]
            # image = (load_image(raw_images_dir + p)*255).astype(np.uint8) # This post-processing did not result in good images.
            image = auto_post_process_image(raw_images_dir + p)
            Image.fromarray(image).resize(h_parameters['resize']).save(processed_images_dir + image_id + '.png', format='png')
            print(f'Saved image to: {processed_images_dir + image_id + ".png"}')

    # Main train loop, for each epoch we train the model on all images.
    images_paths = os.listdir(processed_images_dir)
    for epoch in range(h_parameters['epochs']):
        epoch_loss = []
        batch_inputs, batch_targets = [], []
        # Looping over all images in the processed image directory.
        for i, p in enumerate(images_paths):
            print('Training on image:', p)
            image_id = p.split('.')[0]
            targets = targets_per_image[image_id]
            # Load the image and convert it to a tensor.
            image_np_array = cv2.imread(os.path.join(processed_images_dir, p))
            image_tensor = th.from_numpy(image_np_array).unsqueeze(0).permute(0, 3, 1, 2).div(255.0)

            # Add the image and targets to the batch.
            batch_inputs.append(image_tensor)
            batch_targets.append(targets)

            # If we have a full batch, train the model
            if len(batch_inputs) % h_parameters['batch_size'] == 0:
                # Pull the images trough the GenISP model and ObjectDetector.
                gen_isp_outputs = gen_isp(batch_inputs)

                # The Object Detector we used outputs the losses in training mode.
                object_detector_losses = object_detector(gen_isp_outputs, batch_targets)

                # Only a training step is performed on the GenISP model.
                gen_isp.optimizer.zero_grad()

                # Classification loss + Bounding box regression loss.
                total_loss = object_detector_losses['classification'] + object_detector_losses['bbox_regression']
                total_loss.backward()
                gen_isp.optimizer.step()
                epoch_loss.append(total_loss.item())
                batch_inputs, batch_targets = [], [] # Reset the batch.

        # Print the loss for the epoch after having looped over all images.
        print(f'{epoch} | Epoch loss: {np.mean(epoch_loss)}+/-{np.std(epoch_loss)}')

        # Save the gen_isp outputs after each epoch.
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
