# Foundations of CNNs : convolutional neural networks
Computer vision use `convolution layers` instead of the `fully connected layers`

Three kinds of layers in a CNN:
- `CONV` Convolution
- `POLL` Pooling (Max vs Average)
- `FC` Fully connected

### Notations
```
#  Hyperparameters
f[l] = filter size
p[l] = padding	# Default is zero
s[l] = stride
n_c[l] = number of filters
n[l] = (n[l-1] + 2p[l] - f[l] / s[l]) + 1
# Objects & their sizes
Input:  (n[l-1], n[l-1], n_c[l-1])	Or	 (n_H[l-1], n_W[l-1], n_c[l-1])
Output: (n[l], n[l], n_c[l])	Or	 (n_H[l], nW[l], n_c[l])

Filters: (f[l], f[l], n_c[l-1])

Activations: a[l] is (n_H[l], n_W[l], n_c[l])
		     A[l] is (m, n_H[l], n_W[l], n_c[l])   # In batch or minibatch training

Weights: (f[l], f[l], n_c[l-1], n_c[l])
Bias:  (1, 1, 1, n_c[l])
```

## Why convolutions?
Two main advantages of Convs are:
  - Parameter sharing.
    - A feature detector (such as a vertical edge detector) that's useful in one part of the image is probably useful in another part of the image.
  - Sparsity of connections.
    - In each layer, each output value depends only on a small number of inputs which makes it translation invariance.
