## Introduction

### Description of the original paper
### Data

## Reproduction

### Our approach
### Methods
#### Pre-processing
Preprocessing pipeline consists of two parts, namely packing and Color Space Transformation. 

The implementation of packing is shown as follows.
'''
    raw_image = raw.raw_image.astype(np.int32)
    packed_image = np.zeros((int(raw_image.shape[0] / 2), int(raw_image.shape[1] / 2), 4), dtype=np.int32)
    packed_image[:, :, 0] = raw_image[0::2, 0::2]  # R Left top
    packed_image[:, :, 1] = raw_image[0::2, 1::2]  # G Right top
    packed_image[:, :, 2] = raw_image[1::2, 0::2]  # G Left bottom
    packed_image[:, :, 3] = raw_image[1::2, 1::2]  # B Right bottom
'''
The packing is carried out in the way that a pixel of the packed image is a vector of four elements which are the four square neighbouring pixels of the original raw image. After packing, the two green channels are averaged and returned as an RGB image.

Colour Space Transformation is to convert images' colour space into device independent using the matrix specific to the device. The matrix can be found as one of fields of a raw object after loading an image. By doing this, the model achieves a higher level of generalizability. 

#### GEN-ISP
#### Training
### Results

## Conclusions
