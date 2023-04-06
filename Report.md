## Introduction

### Description of the original paper

### Data
The dataset consists of 7K raw images collected
using two cameras, Sony RX100 (3.2K images) and
Nikon D750 (4.0K), and bounding box annotations of
people, bicycles and cars. The images have been taken in different low-light conditions: ranging from pitch dark to less challenging conditions with artificial lighting.
Authors have made the dataset publicly available to allow benchmarking of future methods targeting object detection in low-light conditions.

#### What is raw image format?
A raw image format refers to the unprocessed image data 
captured by a digital camera's sensor.


#### Why use raw image format?
TODO explain this:
"As showed by Hong et al. [7], detectors us-
ing raw sensor data perform significantly better than de-
tectors using sRGB data processed by a traditional ISP
pipeline."
n contrast with these methods, we propose a neural ISP that
adapts raw image data into representation optimal for machine cognition so that a pre-trained object detector can be
used without any need for fine-tuning or re-training."
## Reproduction
![img.png](report_resources/screenshot_pipeline.png)
### Our approach
### Methods
#### Pre-processing
Preprocessing pipeline consists of two parts, namely packing and Color Space Transformation. 

The implementation of packing is shown as follows.
```
    raw_image = raw.raw_image.astype(np.int32)
    packed_image = np.zeros((int(raw_image.shape[0] / 2), int(raw_image.shape[1] / 2), 4), dtype=np.int32)
    packed_image[:, :, 0] = raw_image[0::2, 0::2]  # R Left top
    packed_image[:, :, 1] = raw_image[0::2, 1::2]  # G Right top
    packed_image[:, :, 2] = raw_image[1::2, 0::2]  # G Left bottom
    packed_image[:, :, 3] = raw_image[1::2, 1::2]  # B Right bottom
```
The packing is carried out in the way that a pixel of the packed image is a vector of four elements which are the four square neighbouring pixels of the original raw image. After packing, the two green channels are averaged and returned as an RGB image.

Colour Space Transformation is to convert images' colour space into device independent using the matrix specific to the device. The matrix can be found as one of fields of a raw object after loading an image. By doing this, the model achieves a higher level of generalizability.

#### GEN-ISP
#### Training
### Results

## Conclusions
