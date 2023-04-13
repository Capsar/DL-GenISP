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
<p float="left">
  <img src="data/results/paper_figure1.jpeg" width="32%" />
  <img src="data/results/paper_figure3.jpeg" width="66%" /> 
</p>

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

Colour Space Transformation is to convert images' colour space into device independent using the matrix specific to the device. The matrix can be found as one of the fields of a raw object after loading an image. By doing this, the model achieves a higher level of generalizability.

Although we implemented this part in the beginning, the reproduction was not an easy task due to its poor explanation, especially image packing, which took us some time to figure out how it works. Additionally, due to its low performance, we decided to use the RawPy build in the function of "rawpy.postprocess" instead of this preprocessing pipeline. 

#### GEN-ISP
The main body of Gen ISP consists of mainly three components, ConvWB, ConvCC and Shallow ConvNet, as shown in the image above.

ConvWB is implemented to adjust global illumination levels and white balance of the image, while ConvCC is to map the colour space so that is optimal for a shallow ConvNet at the end of the entire pipeline. The image is resized and passed to Image-to-Parameter modules, while are three convolutions of tensors with different sizes with Leaky Rectified Linear Unit and Max pooling in between and followed by adaptive averaging pooling and MLP at the end. 

##### ConvWB
$$
\left[\begin{array}{l}
R^{\prime} \\
G^{\prime} \\
B^{\prime}
\end{array}\right]=\left[\begin{array}{ccc}
w_{11} & 0 & 0 \\
0 & w_{22} & 0 \\
0 & 0 & w_{33}
\end{array}\right]\left[\begin{array}{l}
R \\
G \\
B
\end{array}\right] .
$$

##### ConvCC
$$
\left[\begin{array}{l}
R^{\prime} \\
G^{\prime} \\
B^{\prime}
\end{array}\right]=\left[\begin{array}{lll}
c_{11} & c_{12} & c_{13} \\
c_{21} & c_{22} & c_{23} \\
c_{31} & c_{32} & c_{33}
\end{array}\right]\left[\begin{array}{l}
R \\
G \\
B
\end{array}\right] .
$$

After ConvWB and ConvCC, it is passed to a non-linear image enhancement by a shallow ConvNet, which are also a sequence of two convolutions, where there are Instance normalizations and a Leaky Rectified Linear Unit in between.

At the end, the entire pipeline are implemented as below. 

```
class GenISP(th.nn.Module):

    def __init__(self):
        super().__init__()
        # minimal pre-processing pipeline packing and color space transformation

        # 2-step color processing stage realized by image-to-parameter modules: ConvWB and ConvCC
        self.image_to_parameter = th.nn.Sequential(
            th.nn.Conv2d(3, 16, kernel_size=7, padding=3), th.nn.LeakyReLU(), th.nn.MaxPool2d(kernel_size=2),
            th.nn.Conv2d(16, 32, kernel_size=5, padding=2), th.nn.LeakyReLU(), th.nn.MaxPool2d(kernel_size=2),
            th.nn.Conv2d(32, 128, kernel_size=3, padding=1), th.nn.LeakyReLU(), th.nn.MaxPool2d(kernel_size=2),
            th.nn.AdaptiveAvgPool2d(1),
            th.nn.Flatten(1),
        )

        self.conv_wb = th.nn.Sequential(
            Resize((256, 256)),
            deepcopy(self.image_to_parameter),
            th.nn.Linear(128, 3),
            Diagonalize(),
        )

        self.conv_cc = th.nn.Sequential(
            Resize((256, 256)),
            deepcopy(self.image_to_parameter),
            th.nn.Linear(128, 9),
            th.nn.Unflatten(1, (3, 3))
        )

        # A non-linear local image enhancement by a shallow ConvNet
        self.shallow_conv_net = th.nn.Sequential(th.nn.Conv2d(3, 16, kernel_size=3, padding=1), th.nn.InstanceNorm2d(16), th.nn.LeakyReLU(),
                                                 th.nn.Conv2d(16, 64, kernel_size=3, padding=1), th.nn.InstanceNorm2d(64), th.nn.LeakyReLU(),
                                                 th.nn.Conv2d(64, 3, kernel_size=1))

        self.optimizer = th.optim.Adam(self.parameters(), lr=1e-2)

    def forward(self, batch):
        """
        :param batch: batch of images
        :return: enhanced images
        """
        output = []
        for image in batch:
            wb_matrix = self.conv_wb(image)
            image = th.matmul(image.permute(0, 2, 3, 1), wb_matrix).permute(0, 3, 1, 2)
            cc_matrix = self.conv_cc(image)
            image = th.matmul(image.permute(0, 2, 3, 1), cc_matrix).permute(0, 3, 1, 2)
            x = self.shallow_conv_net(image)
            output.append(x.squeeze(0))
        return output
```
Source: https://github.com/Capsar/2022_Q3---DL-GenISP/blob/main/group52/gen_isp.py

Regarding the reproducibility of this module, the explanation of ConvWB and ConvCC was confusing as the only place where it shows the flow of the pipeline was Figure 3 (the image above). An explanation of the flow should have taken place in the paragraph body and how these modules were integrated into the entire pipeline.   
#### Training
In order to train, we have used RetinaNet model available on (https://github.com/pytorch/vision/blob/main/torchvision/models/detection/retinanet.py). This models will be compared with the output of the current network and calculate losses.

As for the loss functions, the paper defined loss as the sum of classification error and regression error. The classification error is implemented by alpha-balanced focal loss wheras the regression loss is by smooth-L1 loss. Both of the implemetations were already there within the RetinaNet model, thus we reused them to compute the losses. 


### Results

## Conclusions
