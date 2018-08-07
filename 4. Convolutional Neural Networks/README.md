# Foundations of CNNs : convolutional neural networks
Computer vision use `convolution layers` instead of the `fully connected layers`

Three kinds of layers in a CNN:
- `CONV` Convolution
- `POLL` Pooling (Max vs Average)
- `FC` Fully connected



Concepts:
## Padding
Problems with convolutions are:
- Shrinks output.
- throwing away a lot of information that are in the edges.
As a solution, we use padding and insert `p` rows/columnsin top, bottom, left and right of the input image before convolution.


## Stride --> strided convolution
When we are making a strided convolution operation we used `s` to tell us the number of pixels we will jump when we are convolving filter/kernel.

General rule: if a matrix `(n, n)` is convolved with a `(f, f)` filter/kernel and padding `p` and stride `s` it gives a `(n+2p-f)/s + 1,(n+2p-f)/s + 1` matrix.

`Same convolution` is a convolution with a padding so that output size is the same as the input size. Its given by the equation: `p = (n*s - n + f - s) / 2` and `when s = 1 ==> P = (f-1) / 2
`

Example of a convolutional layer:
- Input image: 6x6x3 # a0
- 10 Filters: 3x3x3 #W1
- Result image: 4x4x10 #W1a0
- Add b (bias) with 10x1 will get us : 4x4x10 image #W1a0 + b
- Apply RELU will get us: 4x4x10 image #A1 = RELU(W1a0 + b)
- In the last result p=0, s=1
- Hint number of parameters here are: (3x3x3x10) + 10 = 280


## Pooling

The max pooling is saying, if the feature is detected anywhere in this filter then keep a high number. But the main reason why people are using pooling because its works well in practice and reduce computations.
Max pooling has no parameters to learn.


### Notations
#### Hyperparameters

```
f[l] = filter size
p[l] = padding	# Default is zero
s[l] = stride
n_c[l] = number of filters used in the convolution
n[l] = (n[l-1] + 2p[l] - f[l] / s[l]) + 1
```
#### Objects & their sizes
```
Input:  (n[l-1], n[l-1], n_c[l-1])	Or	 (n_H[l-1], n_W[l-1], n_c[l-1])
Output: (n[l], n[l], n_c[l])	Or	 (n_H[l], nW[l], n_c[l])

Filters: (f[l], f[l], n_c[l-1])

Activations: a[l] is (n_H[l], n_W[l], n_c[l])
		     A[l] is (m, n_H[l], n_W[l], n_c[l])   # In batch or minibatch training

Weights: (f[l], f[l], n_c[l-1], n_c[l])
Bias:  (1, 1, 1, n_c[l])
```
Parameters to train:
- if you are using regular NN (not convolutional network): number of parameters is ` height_image * width_image * number_of_channels * number_of_neurons + number_of_neurons ` (summing weights and biases of every neuron)
- if you are using a convolutional network (not a regular NN): number of parameters is
` height_filter * width_filter * number_of_channels * number_of_filters + number_of_filters `
For an input RGB image, ` number_of_channels =3`


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
Very very deep NNs are difficult to train because of vanishing and exploding gradients problems.

We use `skip connection` as a solution which makes you take the activation from one layer and suddenly feed it to another layer even much deeper in NN. That allows you to train large NNs even with layers greater than 100!

Unlike normal networks, these residual networks can go deeper without hurting the performance.
- Plain Networks - the theory tell us that if we go deeper we will get a better solution to our problem, but because of the vanishing and exploding gradients problems the performance of the network suffers as it goes deeper
- Residual Network we can go deeper as we want now.


Let's analyse following NN network containing a residual block:
`X --> Big NN --> a[l] --> Layer1 --> Layer2 --> a[l+2]`

```
a[l+2] = g( z[l+2] + a[l] )
	     = g( W[l+2] a[l+1] + b[l+2] + a[l] )
```

## Network in Network and 1 X 1 convolutions
Replace `fully connected layers` with `1 x 1 convolutions` as Yann LeCun believes they are the same.


## Inception network motivation
When you design an inception network, instead of choosing _which layers_ to pick among some possible ones (A 3 x 3 Conv or 5 x 5 Conv or maybe a max pooling layer, etc), you decide to **use all of them** at once!

An inception network consist of concatenated blocks of the Inception module.

## Using Open-Source Implementation
It turns out that a lot of these NN are difficult to replicated.
When reading a paper and wanting to replicate the model, look first for an open source implementation of this paper. It will save you time and you'll the model pre-trained with the good weights/parameters (learned with higher computational resources than yours).

## Data augmentation


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

## Landmark Detection

## Sliding Windows

## Bounding Box & YOLO algorithm
YOLO is a state-of-the-art object detection model that is fast and accurate

## Non-max Suppression
Intersection Over Union between bouding boxes.

Non-max suppression algorithm:
1. Lets assume that we are targeting one class as an output class.
2. Y shape should be `[Pc, bx, by, bh, hw]` Where Pc is the probability if that object occurs.
3. Discard all boxes with `Pc < 0.6`
4. While there are any remaining boxes:
	- Pick the box with the largest `Pc` Output that as a prediction.
	- Discard any remaining box with `IoU > 0.5`

If there are multiple classes/object types c you want to detect, you should run the Non-max suppression c times, once for every output class.

## Anchor Boxes
Specialize your algorithm to detect shapes that spans a variety of shapes covering the types of objects you seem to detect frequently.
- hand made
- generated using k-means

---------------
# Special applications: Face recognition & Neural style transfer
