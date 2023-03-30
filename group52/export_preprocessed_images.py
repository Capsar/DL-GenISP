import os

import rawpy
from rawpy._rawpy import ColorSpace, FBDDNoiseReductionMode
from PIL import Image


def auto_post_process_image(path):
    print('Loading image: {}'.format(path))
    with rawpy.imread(path) as raw:
        post_processed_image = raw.postprocess(half_size=True, fbdd_noise_reduction=FBDDNoiseReductionMode.Full)
        return post_processed_image


def main():
    data_dir = '../data/our_sony/'
    raw_images_dir = data_dir + 'raw_images/'
    processed_images_dir = data_dir + 'processed_images/'

    images_paths = os.listdir(raw_images_dir)
    for p in images_paths:
        image_id = p.split('.')[0]
        image = auto_post_process_image(raw_images_dir + p)
        Image.fromarray(image).save(processed_images_dir + image_id + '.png', format='png')


if __name__ == '__main__':
    main()
