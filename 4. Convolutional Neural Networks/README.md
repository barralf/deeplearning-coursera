# Foundations of CNNs : convolutional neural networks
## Computer vision
Computer vision applications: self-driving, face recognition, arts

Computer vision use `convolution layers` instead of the `fully connected layers` to optimize time computations.

Three kinds of layers in a CNN:
- `CONV` Convolution
- `POLL` Pooling (Max vs Average)
- `FC` Fully connected

## Convolution & edge detection example
Concepts illustrated on a simple 2D case.

## Padding
Problems with convolutions are:
- Shrinks output.
- throwing away a lot of information that are in the edges.

As a solution, we use padding and insert `p` rows/columns in top, bottom, left and right of the input image before convolution.


## Stride & Strided Convolution
When we are making a strided convolution operation, we use `s` to tell the number of pixels we will jump when we are convolving filter/kernel.

**General rule:** if a matrix `(n, n)` is convolved with a `(f, f)` filter/kernel and padding `p` and stride `s` it gives a `( (n+2p-f)/s + 1, (n+2p-f)/s + 1 )` matrix.

**Same convolution** is a convolution with a padding so that output size is the same as the input size. Its given by the equation: `p = (n*s - n + f - s) / 2` and `when s = 1 ==> P = (f-1) / 2`

Example of a convolutional layer:
- Input image:` 6x6x3 # a0`
- 10 Filters: `3x3x3 #W1`
- Result image: `4x4x10 #W1a0`
- Add b (bias) with 10x1 will get us : `4x4x10 image #W1a0 + b`
- Apply RELU will get us: `4x4x10 image #A1 = RELU(W1a0 + b)`
- In the last result `p=0, s=1`
- Hint number of parameters here are: `(3x3x3x10) + 10 = 280`


## Convolution over volumes
`n_c`: number of filters
`( (n+2p-f)/s + 1, (n+2p-f)/s + 1 , n_c )` matrix
### Notations
#### Hyperparameters

```
f[l] = filter size
p[l] = padding	# Default is zero
s[l] = stride
n_c[l] = number of filters used in the l-th convolutional layer
n[l] = (n[l-1] + 2p[l] - f[l] / s[l]) + 1
```
#### Objects & their sizes
```
Input:  (n[l-1], n[l-1], n_c[l-1])	Or	 (n_H[l-1], n_W[l-1], n_c[l-1])
Output: (n[l], n[l], n_c[l])	Or	 (n_H[l], n_W[l], n_c[l])

Filters: (f[l], f[l], n_c[l-1])

Activations: a[l] is (n_H[l], n_W[l], n_c[l])
		     A[l] is (m, n_H[l], n_W[l], n_c[l])   # In batch or minibatch training

Weights: (f[l], f[l], n_c[l-1], n_c[l])
Bias:  (1, 1, 1, n_c[l])
```
**Parameters to train**: A REVOIR
- if you are using regular NN (not convolutional network): number of parameters is `n_[l]*n[l-1]+n[l]`(summing weights and biases of every neuron)
- if you are using a convolutional network (not a regular NN): number of parameters is
` height_filter * width_filter * number_of_channels * number_of_filters + number_of_filters ` <=> `f[l]* f[l]*n_c[l-1]|*n_c[l] + n_c[l]`
For an input RGB image, ` number_of_channels =3`

## Pooling
The **max pooling** is if the feature is detected anywhere in this filter then keep a high number. But the main reason why people are using pooling is because it works well in practice and reduce computations.

We can also use **average pooling**.

**Pooling**  reduce `n_H`, `n_W`, but not `n_C`. Also it has no parameters to learn and


## Why convolutions?
Two main advantages of Convs are:
  - **Parameter sharing**.
    - A feature detector (such as a vertical edge detector) that's useful in one part of the image is probably useful in another part of the image.
  - **Sparsity of connections**.
    - In each layer, each output value depends only on a small number of inputs which makes it translation invariance.


We learned about Conv layer, pooling layer, and fully connected layers. It turns out that computer vision researchers spent the past few years on how to put these layers together.

-------

# Deep convolutional models: case studies
Case studies on following famous Conv Nets: LeNet-5, AlexNet, and VGG.

## Residual Networks (ResNets)
Very very deep NNs are difficult to train because of vanishing and exploding gradients problems. Fortunately Residual Networks, introduced by He et al., train much deeper networks than were previously practically feasible with normal NN also called Plain Networks.

We use `skip connection` as a solution which makes you take the activation from one layer and suddenly feed it to another layer even much deeper in NN. That allows you to train large NNs even with layers greater than 100!

Unlike normal networks, these residual networks can go deeper without hurting the performance.
- **Plain Networks** - the theory tell us that if we go deeper we will get a better solution to our problem, but because of the vanishing and exploding gradients problems the performance of the network suffers as it goes deeper
- **Residual Networks** we can go deeper as we want now.

Plain deep network do not always reduce training error, that's why we need ResNet.

Let's analyse following NN containing a residual block:
`X --> Big NN --> a[l] --> Layer1 --> Layer2 --> a[l+2]`

```
a[l+2] = g( z[l+2] + a[l] )
	     = g( W[l+2] a[l+1] + b[l+2] + a[l] )
```

- identity block

- convolutional block

Using skip connection helps the gradient to propagate and thus help use to train much deeper networks.


Details of thie ResNet-50 model of the assessment: are:

	- Zero-padding pads the input with a pad of (3,3)
	- Stage 1:
		- The 2D Convolution has 64 filters of shape (7,7) and uses a stride of (2,2). Its name is "conv1".
		- BatchNorm is applied to the channels axis of the input.
		- MaxPooling uses a (3,3) window and a (2,2) stride.
	- Stage 2:
		- The convolutional block uses three set of filters of size [64,64,256], "f" is 3, "s" is 1 and the block is "a".
		- The 2 identity blocks use three set of filters of size [64,64,256], "f" is 3 and the blocks are "b" and "c".
	- Stage 3:
		- The convolutional block uses three set of filters of size [128,128,512], "f" is 3, "s" is 2 and the block is "a".
		- The 3 identity blocks use three set of filters of size [128,128,512], "f" is 3 and the blocks are "b", "c" and "d".
	- Stage 4:
		- The convolutional block uses three set of filters of size [256, 256, 1024], "f" is 3, "s" is 2 and the block is "a".
		- The 5 identity blocks use three set of filters of size [256, 256, 1024], "f" is 3 and the blocks are "b", "c", "d", "e" and "f".
	- Stage 5:
		- The convolutional block uses three set of filters of size [512, 512, 2048], "f" is 3, "s" is 2 and the block is "a".
		- The 2 identity blocks use three set of filters of size [512, 512, 2048], "f" is 3 and the blocks are "b" and "c".
	- The 2D Average Pooling uses a window of shape (2,2) and its name is "avg_pool".
	- The flatten doesn't have any hyperparameters or name.
	- The Fully Connected (Dense) layer reduces its input to the number of classes using a softmax activation. Its name should be 'fc' + str(classes).

Very deep Residual Networks are built **by stacking these blocks together**.

## Network in Network and 1 X 1 convolutions
Replace `fully connected layers` with `1 x 1 convolutions` as Yann LeCun believes they are the same.

You can use a 1x1 convolutional layer to reduce `n_C` but not `n_H`, `n_W`.


## Inception network motivation
When you design an inception network, instead of choosing _which layers_ to pick among some possible ones (A 3 x 3 Conv or 5 x 5 Conv or maybe a max pooling layer, etc), you decide to **use all of them** at once!

An inception network consist of concatenated blocks of the Inception module.

## Using Open-Source Implementation
It turns out that a lot of these NN are difficult to replicated.
When reading a paper and wanting to replicate the model, look first for an open source implementation of this paper. It will save you time and you'll the model pre-trained with the good weights/parameters (learned with higher computational resources than yours).

## Data augmentation

- Data augmentation can be a good technique to increase the performance of deep NN.

- Example of data augmentation for computer vision: mirroring, random cropping, rotation, shearing, local warping, color shifting.

- We can implement **distorsions** on minibatches _**during training**_  using a different CPU thread.

## Ensembling and multi-cropping

**Ensembling**: train several networks independently and average the outputs. Merging down some classifiers.

-------
# Object detection

## Object Localization

**Image Classification**:
Classify an image to a specific class. The whole image represents one class. We don't want to know exactly where are the object. Usually only one object is presented.

**Classification with localization**:
Given an image we want to learn the class of the image and where are the class location in the image. We need to detect a class and a rectangle of where that object is. Usually only one object is presented.

**Object detection**:
Given an image we want to detect all the object in the image that belong to a specific classes and give their location. An image can contain more than one object with different classes.

Note: Also Semantic Segmentation and Instance Segmentation.

#### Classification with localization
`y=[Pc, bx, by, bh, bw, C1, C2, C3, C4]`

#### Loss functions
- logistic regression for `Pc`
- likelihood loss for `classes`
- squared error for `bouding boxes`

## Landmark Detection
If you want to output some points for instance for face recognition (corners of eyes, nose) or for skeleton detection.

## Sliding Windows

## Bounding Box & YOLO algorithm
YOLO is a **state-of-the-art object detection model** that is fast and accurate.

## Non-max Suppression
Intersection Over Union between bouding boxes.

**Non-max suppression algorithm**:
1. Lets assume that we are targeting one class as an output class.
2. Y shape should be `[Pc, bx, by, bh, bw]` Where Pc is the probability if that object occurs.
3. Discard all boxes with `Pc < 0.6`
4. While there are any remaining boxes:
	- Pick the box with the largest `Pc` Output that as a prediction.
	- Select only one box when several boxes overlap with each other and detect the same class: `IoU > 0.5`


If there are `c` different class/object types you want to detect, you should run the Non-max suppression `c` times, once for every output class.

## Anchor Boxes
Specialize your algorithm to detect shapes that spans a variety of shapes covering the types of objects you seem to detect frequently.
- anchor boxes **hand made**
- anchor boxes **generated using k-means**

---------------
# Special applications: Face recognition & Neural style transfer

Face Verification vs Face Recognition

## One short learning
We use tau `T` as a threshold for d:
If `d( img1, img2 ) <= T` Then the faces are the same.

## Siamese networks
Siamese networks (similar to word2vec). You create embeddings vectors of your images and compute the Euclidean norme.

2 ways to learn the parameters of a conv net for face recognition:
- Triplet Loss
- Similarity function

## Triplet Loss
Triplet loss is one of the loss functions we can use to solve the similarity in a siamese network. (compared to gradient descent and other optimizers in a usual NN). It will train a neural network to learn an encoding of a face image.

`L(A, P, N) = max (||f(A) - f(P)||^2 - ||f(A) - f(N)||^2 + alpha , 0)`

## Similarity function
`Y' = Sigmoid ( sum_k w_k | f(x(i)_k) - f(x(j)_k) | + b)`
with Manhattan distance (could use Euclidean or Ki square similarity instead)

## Neural style transfer

Neural style transfer takes a content image `C` and a style image `S` and generates the content image `G` with the style of style image.

## Cost function

Given a content image `C`, a style image `S`, and a generated image `G`:
`J(G) = alpha * J(C,G) + beta * J(S,G)`

## Content cost function

## Style cost function
Style is defined as correlation between activations across channels.

The correlation of style image channels should appear in the generated image channels.


**Grim matrix**

Formula


The style of an image can be represented using the Gram matrix of a hidden layer's activations. However, we get even better results combining this representation from multiple different layers. This is in contrast to the content representation, where usually using just a single hidden layer is sufficient.

Minimizing the style cost will cause the image  GG  to follow the style of the image  S.

The total cost is a linear combination of the content cost  Jcontent(C,G)Jcontent(C,G)  and the style cost  Jstyle(S,G)Jstyle(S,G)
`alpha`  and  `beta`  are hyperparameters that control the relative weighting between content and style.

Extra tips:

- Steps to be made if you want to create a tensorflow model for neural style transfer:
	- Create an Interactive Session.
	- Load the content image.
	- Load the style image
	- Randomly initialize the image to be generated
	- Load the VGG16 model
	- Build the TensorFlow graph:
	- Run the content image through the VGG16 model and compute the content cost
	- Run the style image through the VGG16 model and compute the style cost
	- Compute the total cost
	- Define the optimizer and the learning rate
	- Initialize the TensorFlow graph and run it for a large number of iterations, updating the generated image at every step.



**Wrap up**:
- Neural Style Transfer is about generating artistic images given a content image C and a style image S. All this by building a model in which the optimization algorithm updates the pixel values rather than the neural network's parameters.
- It uses representations (hidden layer activations) based on a pretrained ConvNet.
- The content cost function is computed using one hidden layer's activations.
- The style cost function for one layer is computed using the Gram matrix of that layer's activations. - The overall style cost function is obtained using several hidden layers.
- Optimizing the total cost function results in synthesizing new images.

## 1D & 3D Generalizations

Conv nets can work with 1D and 3D data as well. Input shape has to be changed "mutatis mutandis".

- **1D data** comes from a lot of resources such as waves, sounds, heartbeat signals.
Usually we use Recurrent Neural Network RNN for 1D data..
- **3D data** like CT scan

Random:

(1) The cans are round, so that means the bounding boxes must be square, i.e. not rectangular. That removes the need to learn one of the bounding box parameters. (2) The cans are always the same size, so the algorithm doesn't need to learn a different bounding box size for every image
