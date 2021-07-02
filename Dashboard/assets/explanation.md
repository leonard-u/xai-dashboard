## What exactly does this app do and what is the benefit of it?
This is an interactive approach for the education of one explainable AI (XAI) method. For this, we have implemented a so-called Gradient-weighted Class Activation Mapping (Grad-CAM), which can be operated interactively.
Our Intention with this dashboard and the code behind it is to make the image classifications based on deep learning easier to interpret.
Interpretability here means that the explanation is understandable for humans.
At first glance, predictions with neural networks are a kind of black box for humans, as the internal mechanisms are unknown and difficult to understand.
Our approach does not only enable a better understanding of how neural networks work but also helps to diagnose and improve them [[1]].


## Target users
This app is for students or professionals, who want to understand the principle, the inner working, the benefits, and potentially the drawbacks of a Grad-CAM.
The app was designed in such a way that it should also enable people without prior knowledge to understand the predictions in combination with the implemented Grad-CAM.
The [keras documentation](https://doi.org/10.1109/CVPR.2016.90) offers a very good introduction to implementing a Grad-CAM using python.

## Description of Grad-CAM

Grad-CAM is a technique for producing visual explanations for decisions from CNN-based models. 
The main components of this app are the python libraries TensorFlow, Keras, and Dash.
TensorFlow forms the framework for this. Keras serves as an interface to TensorFlow, and Dash is used to display the user interface and interaction. 

TensorFlow 2.X implementation of Grad-CAM is an approach for model explainability which produces a heatmap of which regions of an image contributed strongly towards the final prediction.
This leads to making neural networks more transparent. 
Grad-CAM extends the applicability of the CAM method by including gradient information.
The Grad-CAM uses the gradients of each target concept, which flow into the last convolutional layer, in order to create a localization map in which the important regions for predicting an image are highlighted.
Specifically, the gradient of the loss function in relation to the last folding layer determines the weight for each of the corresponding feature maps.
The further steps consist of calculating the weighted sum of the activations and then upsampling the result to the image size in order to display the original image with the heatmap obtained.
As already mentioned in the introduction, the Grad-CAM also helps to increase the trustworthiness of neural networks. This also enables biases to be found in datasets.[[2]]
According to the guiding principle: A true AI system should not only be intelligent but also be able to reason about its beliefs and actions for humans to trust it. 

**Working principle Grad-CAM:**[[2]]

![img_1.png](http://gradcam.cloudcv.org/static/images/network.png)


## How to use this app?
The "information-seeking mantra" by Ben Shneidermann is used for visualization and at the same time for handling the app.
First, an overview is given, and details can be called up-on demand using various filters.
The app is divided into three parts and the user must follow these three steps:

**Step 1)**

The left column houses the function for uploading an image either via a simple selection or via drag and drop.
Furthermore, the architecture (e.g. ResNet50 or VGG16) of the neural network can also be defined there.
The selected architectures are pre-trained on the ImageNet dataset. 
This means that the weights of this dataset are used for the later predictions.
ImageNet is a large visual database designed for use in visual object recognition software research. 
More than 14 million images have been hand-annotated by the project to indicate what objects are pictured and in at least one million of the images, bounding boxes are also provided.
ImageNet contains more than 20,000 categories with a typical category, such as "balloon" or "strawberry", consisting of several hundred images.
The choice of architecture can have a major influence on the generated heatmap.

All previous entries can be reset using the reset button.

**Step 2)**

After uploading the image, it will be shown as original in the middle column.
This image is permanently saved and is spared from changes, so it remains in the original.
At the same time, the predictions based on the previously selected architecture of the neural network are displayed.
Furthermore, the user has the option of limiting the number of predictions via the slider below.
This allows the user to decide how many predictions are displayed. The range is from 1-10.

**Step 3)**

Step 3 is about the implementation of the Grad-CAM. There are three options to choose from: Two filter options and a slider.
Depending on the selections, these are then immediately applied to the display of the Grad-CAM above.

First, the Grad-CAM style can be selected.
In addition to the conventional Grad-CAN, there is also the option of selecting the so-called counterfactual explanation Grad-CAM, for example.
A map with regions is produced here, that would lower the network's confidence in its prediction.
This is useful when two competing objects are present in the image. 
We can produce a "counterfactual" image with these regions masked out, which should give a higher confidence in the original prediction.
Using the slider, which is used to define the alpha channel, the strength of the color which is used for Grad-CAM can be varied.
Another option is the heatmap style, which can be used to influence the color scheme of the displayed heatmap in the Grad-CAM image.


## Description of the implemented architectures
The available architectures are very different in terms of focus. a comparatively high level of accuracy can be achieved using VGG16 and ResNet50. 
Xception and MobileNetV2 focus on lightweight and are particularly suitable for low-resource devices such as edge devices or mobile phones.

**VGG16:**

VGG16 is a convolutional neural network model proposed by K. Simonyan and A. Zisserman [[3]]. 
The model achieves 92.7 % top-5 test accuracy in ImageNet.
The most unique thing about VGG16 is that instead of having a large number of hyper-parameter they focused on having convolution layers of a 3x3 filter with a stride 1 and always used the same padding and max pool layer of 2x2 filter of stride 2.
The 16 in VGG16 refers to it has 16 layers that have weights. 
This network is a pretty large network and it has about 138 million (approx) parameters.
One of the drawbacks of the VGGet is the very long time it takes to train a model.
The network architecture weights themselves are quite large

**ResNet50:**

ResNet-50 is a convolutional neural network that is 50 layers deep. ResNet stands for Residual Network.
This model was the winner of ImageNet challenge in 2015.
The fundamental breakthrough with ResNet was it allowed us to train extremely deep neural networks.
Prior to ResNet training very deep neural networks were difficult due to the problem of vanishing gradients.

ResNet first introduced the concept of skip connection.
The ResNet-50 model consists of 5 stages each with a convolution and Identity block. 
Each convolution block has 3 convolution layers and each identity block also has 3 convolution layers. 
The ResNet-50 has over 23 million trainable parameters. [[4]]

**MobileNetV2:**

MobileNetV2 is very similar to the original MobileNet, except that it uses inverted residual blocks with bottlenecking features. 
It has a drastically lower parameter count than the original MobileNet.

MobileNetV2, by Google, is suitable for mobile devices, or any devices with low computational power. [[5]]

**Xception:**

Xception, which architecture was created by Google, stands for Extreme version of Inception.
Instead of conventional convolution layers, separable convolution layers are used.
This technique reduces the number of connections and makes the model lighter. 
With a modified depthwise separable convolution, it is even better than Inception-v3 for ImageNet.

In this light, a depthwise separable convolution can be understood as an Inception module with a maximally large number of towers. 
This observation leads them to propose a novel deep convolutional neural network architecture inspired by Inception, where Inception modules have been replaced with depthwise separable convolutions. [[6]]

## References
[[1]]: Wojciech Samek, Thomas Wiegand, Klaus-Robert Müller. 2017. Explainable Artificial Intelligence: Understanding, Visualizing and Interpreting Deep Learning Models. arXiv: 1708.08296

DOI: [10.1007/978-3-030-28954-6](https://arxiv.org/abs/1708.08296)

[[2]]: Ramprasaath R. Selvaraju, Abhishek Das, Ramakrishna Vedantam, Michael Cogswell, Devi Parikh, and Dhruv Batra. 2020. Grad-cam: Why did you say that? visual explanations from deep networks via gradient-based localization. arXiv: 1610.02391

DOI: [10.1007/s11263-019-01228-7](https://arxiv.org/ct?url=https%3A%2F%2Fdx.doi.org%2F10.1007%2Fs11263-019-01228-7&v=34fb63fb)

[[3]]: Simonyan Karen, Zisserman Andrew. 2014. Very Deep Convolutional Networks for Large-Scale Image Recognition. arXiv: 1409.1556 

Link: [https://arxiv.org/abs/1409.1556](https://arxiv.org/abs/1409.1556)

[[4]]: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. 2016. Deep Residual Learning for Image Recognition. In CVPR, pages 770-778

DOI: [10.1109/CVPR.2016.90](https://doi.org/10.1109/CVPR.2016.90)

[[5]]: Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen. 2018. MobileNetV2: Inverted Residuals and Linear Bottlenecks. In CVPR, pages 4510-4520

DOI: [10.1109/CVPR.2018.00474](https://doi.org/10.1109/CVPR.2018.00474)

[[6]]: Francois Chollet. 2017. Xception: Deep Learning with Depthwise Separable Convolutions. In CVPR

DOI: [10.1109/CVPR.2017.195](https://doi.org/10.1109/CVPR.2017.195)


------------------------------------------------
**Authors:**
Niklas Groiss, 
Markus Schleicher, 
and Leonard Uhlisch