import cv2
import numpy as np
import rawpy
from rawpy._rawpy import FBDDNoiseReductionMode


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
        packed_image[:, :, 0] = raw_image[0::2, 0::2]  # R Left top
        packed_image[:, :, 1] = raw_image[0::2, 1::2]  # G Right top
        packed_image[:, :, 2] = raw_image[1::2, 0::2]  # G Left bottom
        packed_image[:, :, 3] = raw_image[1::2, 1::2]  # B Right bottom

        # averaged green channel
        averaged_image = np.zeros((packed_image.shape[0], packed_image.shape[1], 3), dtype=np.int32)
        averaged_image[:, :, 0] = packed_image[:, :, 0]  # R
        averaged_image[:, :, 1] = (packed_image[:, :, 1] + packed_image[:, :, 2])  # G
        averaged_image[:, :, 2] = packed_image[:, :, 3]  # B

        # convert color channel
        conversion_matrix = raw.rgb_xyz_matrix

        # Or from packed or from averaged
        # xyz_image = packed_image @ conversion_matrix
        # plt.imshow(xyz_image / 2**13)
        # plt.show()

        xyz_image = averaged_image @ conversion_matrix[0:3, 0:3]
        # plt.imshow(xyz_image / 2**13)
        # plt.show()

        return xyz_image


def auto_post_process_image(path):
    """
    Post-processes the raw image using RawPy.
    :param path: Image path.
    :return: post-processed image.
    """

    print('Loading image: {}'.format(path))
    with rawpy.imread(path) as raw:
        post_processed_image = raw.postprocess(half_size=True, fbdd_noise_reduction=FBDDNoiseReductionMode.Full)
        return post_processed_image


def draw_boxes(boxes, classes, image):
    for box, cls in zip(boxes, classes):
        color = cls[1]
        cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
        cv2.putText(image, cls[0], (int(box[0]), int(box[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, [255,255,255], 2, lineType=cv2.LINE_AA)
    return image
