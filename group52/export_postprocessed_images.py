import os
from PIL import Image
from image_helper import auto_post_process_image


def main():
    """
    Method for post-processing the RAW images and saving them as PNG images.
    :return:
    """
    data_dir = '../data/our_sony/'
    raw_images_dir = data_dir + 'raw_images/'
    processed_images_dir = data_dir + 'processed_images/'

    images_paths = os.listdir(raw_images_dir)
    for p in images_paths:
        image_id = p.split('.')[0]
        image = auto_post_process_image(raw_images_dir + p)
        print(image.shape, type(image), image.dtype)
        Image.fromarray(image).resize((1333, 800)).save(processed_images_dir + image_id + '.png', format='png')


if __name__ == '__main__':
    main()
