# ResNet
Image Classification with Limited Size ResNet
Mini Project 1 Report - Image Classification with Limited Size ResNet
Aditya Adya
New York University



Abstract

INTRODUCTION

With image classification being a fundamental task for cutting edge technology in fields such as cancer treatment and self-driving cars, it is important that these models achieve high accuracy with limited complexity. Nowadays, machine learning models can be integrated in personal devices and applications that can only reserve a limited amount of memory for such models. Large image classifier models are extenuating to train and hard to store, hence we propose the implementation of a Residual Neural Network (ResNet) model for image classification with less than 5 million trainable parameters and show that such models can still achieve high accuracy, while having much lower memory requirements.

METHODOLOGY

Data
The dataset of choice is the CIFAR10 dataset, a widely used image dataset containing 50k training samples and 10k test samples. The samples are 32x32 colored images of objects belonging to 10 different classes, such as dogs, cats, airplanes, and cars.
The models tested were trained without the use of a validation portion, on the entire 50k training samples, plus other samples obtained through data augmentation. The models were tested against the 10k test samples and the training data was organized in batches of 32 samples.

Augmentation
The dataset has been augmented five times, resulting in a total training dataset of 300k samples, including the original images. There were five augmentations of choice, each made on a copy of the original dataset. The inspiration for these augmentations has been taken by a variety of online sources and papers, and the techniques are widely used in the machine learning community. (QUOTE)
The first augmentation of choice was to create a horizontally flipped version of the dataset, just to guarantee that the model would learn all objects symmetrically. For all following four augmentations, the dataset was horizontally flipped at random with a probability of each image getting flipped of 0.5, before applying the other augmentation techniques. The second and third augmentation effectively result in a scaling and a translation. However, in one the image is scaled down, while in the other one it was scaled up. The fourth augmentation was a random rotation of the image by a random angle in between -90 and 90 degrees. The choice of angle range was based on the assumption that the objects would never present themselves upside down, and hence there is no need of rotating the images beyond that range. The fifth and final augmentation was a random change in brightness, contrast, saturation, and hue of the image. Each change was in between 0.5 and 1.5 of the original value.

Training Regiment
The loss function used for the problem was a standard multi-class Cross-Entropy loss function while the optimizer of choice was a vanilla stochastic gradient descent algorithm, with momentum of 0.9 and weight decay of 0.0001. The optimizer used a MultiStep Learning rate scheduler, with learning rate starting at 0.1 and decreasing by a factor of 10 at different epochs (milestones) in the training phase. The following approach was inspired by the following paper (https://arxiv.org/abs/1705.08292), which shows how a momentum based SGD optimizer with a specific learning rate schedule yields higher-quality local minima than the more widely used adaptive techniques such as ADAM and Adagrad. The specific learning rate decay schedule we have used for our models was initially similar to the one used here (https://arxiv.org/abs/1512.03385). The learning rate was decreased by a factor of 10 at 50% and 75% of the training epochs. In the paper, 64k iterations, while our model was trained for 300 epochs. The number of epochs selected was based on an educated guess of the speed at which the loss was decreasing and saturating. Later on we decreased the number of epochs to 150, with learning rate decays at 20, 50, and 125 epochs. Ultimately however, for our highest accuracy models we have used an even tighter schedule. With the previous schedules and epochs, the amount of epochs where the loss was saturated before the next decay milestone was significantly high, meaning the loss would saturate much quicker than the learning rate would decay, sometimes even showing signs of overfitting. Hence, we decided to pick milestones that were few epochs after the loss started saturating. The final schedule has milestones at 15, 25, 35, and 45 epochs, over a total of 50 epochs, starting as always from a learning rate of 0.1 and decreasing to a learning rate of 0.00001. This scheduling gave us the highest performing models we trained.

ARCHITECTURE

For our network architecture, we have used the original ResNet18 architecture as a template. We have not modified any activation function, nor included techniques such as bottlenecks for our experiments. The only aspects we experimented on were the number of residual layers, the number of residual blocks in each layer, the number of channels in each block, and the kernel size in each convolutional layer. The ResNet 18 is an 18 trainable layers residual neural network with 4 residual layers, each containing 2 residual blocks, with each block containing 2 convolutional layers and a skip connection. The 4 residual layers follow an initial convolutional layer and are followed by a fully connected layer. The channels increase from 64 to 512 by a factor of two with each new residual layer. ResNet18 however contains more that 5 million parameters and hence, our modifications to the architecture of the model were directed by the problem of what should we reduce first to decrease the model parameters. Our options were, the number of residual layers, the number of residual blocks, the number of channels, or the kernel sizes. 


Approach I: Reducing Blocks and Kernel Size
The first attempt at modifying the model was reducing the residual block in each of the residual layers, having just one block per layer, with everything else equal. However, this model did not meet our parameter requirement and hence we had to also reduce some other characteristics of the model to make it work. Because we wanted to leave the decrease in channels or residual layers as separate experiments, our choice was to reduce the kernel size of the first convolutional layer in each residual block from a 3x3 to a 1x1 kernel. The decision was made based on the fact that, reducing the kernel size in both convolutional layers of a block would reduce parameter count too significantly. After that we noticed we still had space for a few more blocks in the network and so we added 1 block to the second layer and 2 blocks to the third layer. As to how we chose to assign blocks we were inspired by the following code (https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py) where different ResNet architectures are shown. We had noticed that there is a tendency in these architectures of having more blocks in the second and third layer in particular, or the middle layers more generally. With these choices the resulting model contained about 4.9 million parameters.

Approach II (a): Reducing Layers and Increasing Blocks
Our second approach was eliminating the last residual layer containing convolutional layers with 512 channels, as these are the ones that have the highest parameter count. Eliminating the last layer was sufficient to bring the model far beneath 5 million parameters. Hence, we increased the model size to closely match the requirement by adding residual blocks to the 3 remaining residual layers. The decision to increase block size was due to the impossibility of increasing channels or kernel size in the convolutional block as these choices would have not yielded a model that met our parameter restrictions. The final architecture contains 3 residual blocks in the first residual layer, 5 blocks in the second layer, and 3 in the last layer, bringing our model to about 4.9 million parameters. All other characteristics were kept the same.

Approach II (b): Reducing Layers and Increasing Kernel
In this approach we tried reducing the number of layers similar to the last approach where we got rid of the residual layer,  but instead of increasing the number of blocks, instead we increased the kernel size in the first convolution layer of each block to ‘5’ instead of ‘3’ as in the traditional approach. The parameters were still around 4.9 million.

Approach IIIa: Reducing Channels and Increasing Blocks
For the following approach we have selected to reduce the number of channels in the original model to by half their value, starting at 32 channels instead of 64, and ending with 256. This operation brought parameter count beneath 5 million, and the model was made to reach approximately 4.9 million parameters by adding more blocks in each of the 4 layers. The final architecture saw 3 blocks in layers 1,2, and 4, and 5 clocks in layer 3.

Approach IIIb: Reducing Channels and Increasing Kernel Size
Similar to approach III a we reduced the channel size starting from 32 but instead of adding more block in layers we increased the 


RESULTS AND DISCUSSION


Method
Test Accuracy
Approach I
94.24%
Approach II (a)
95.26%
Approach II (b)
95.1%
Approach III (a)
94.23%
Approach III (b)
93.76%


The model architecture producing the best results was the model obtained through the second approach, which yielded a test accuracy of 95.26%.

From experimenting, we have noticed that the difference in model architecture did not yield widely divergent results. 

From approaches I and II, it can be inferred that, given a target model parameter count, sacrificing a residual layer to increase the number of residual blocks, when the residual blocks are too few, such as in approach I, is preferable. In fact, it seems a rule of thumb to have at least 2 blocks per layer. We imagine that if we could fit at least  2 blocks per layer, in a 4 layer model, that the model would perform better. Furthermore, the reduction in kernel size also seems to have negatively affected model performance. Increasing the number of residual blocks too much, at the expense of some other attribute, will however also negatively impact model performance, as demonstrated by the following paper (https://arxiv.org/pdf/1512.03385.pdf), which shows how deeper ResNets, are more difficult to train and perform worse. Hence approach II seems the more sensible architecture choice given the constraints, and is in fact the one yielding the best results. 
Reducing the number of channels to preserve the amount of residual layers does not seem to be the best choice either. As the model does not perform better than the one in approach 
The most influential factors in increasing model performance were the data and training regiment it received. Data augmentation was by far the most important factor in increasing the model accuracy, bringing most models tested to yield around 90% or more accuracy on testing. The augmentations the data received listed above were the ones that yielded the greatest increase in test accuracy. Surprisingly enough, for the rotation transformation we thought that smaller angle ranges would be more appropriate, however, -90 to 90 degrees was the one that gave us the highest results during testing. We have also thoroughly tested other possible geometric transformations for the images provided in the PyTorch library, but they did not yield a significant contribution. The batch size of 32 was chosen as the model tends to slightly overfit with larger batch sizes.

For the training schedule, we 

CONCLUSION
From the experiments conducted, building a limited size ResNet seems to yield better results when the channels layout, number of residual blocks, and the kernel size are preserved, while sacrificing the number of residual layers. If the number of residual layers start becoming too small, then it is opportune to decrease the number of channels, as can be seen in the architectures of various popular ResNets with fewer than 5 million parameters. (https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py).
Most popular Resnet Architectures use 3x3 kernels and it does not seem to be the case that many successful ResNet or CNN architectures, choose to increase the kernel size.




