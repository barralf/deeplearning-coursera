# Neural Networks and Deep Learning: Overview & Notations
Get an overview of how a Neural Network works in the most general case (as possible). 🤘

Note: a NN with 1 layer & 1 neuron is equivalent to a logistic regression `z = w.T x + b and y = sigmoid(z)`

## Sizes
- `m` is the number of training vectors
- `n_x` is the size of the input vector or input layer
- `n_y` is the size of the output vector or output layer
- `L` is the _total number of layers_ (`L-1` would be then the _number of hidden layers_)
- `n[l]` is the number of neurons/units in the l-th layer (n[0]=n_x and n[L]=n_y)

## Objects
- `x(i)` is the i-th input vector of shape (n_x,1)
- `y(i)` is the i-th output vector of shape (n_y,1)
- `X = [x(1) x(2).. x(m)]` of shape (n_x,m)
- `Y = [y(1) y(2).. y(m)]` of shape (n_y,m)
- `W[l]` is the weight matrix of the l-th layer and of shape (n[l],n[l-1]) (if n[l-1]=1, we use small `w[L]`)
- `b[l]` is the bias vector of the l-th layer and of shape (n[l],1)
- `z[l]` pre-activation of the l-th layer of shape (n[l],1)
- `Z[l]` pre-activation of the l-th layer of shape (n[l],m)
- `g[l]` is the activation function of the l-th layer
- `a[l]` activation of the l-th layer of shape (n[l],1)
- `A[l]` activations of the l-th layer shape (n[l],m)

Convention: `A[l](m)_i = A[layer](example)_entryvector`

## Functions
- Loss Function `L(y',y) = - (y*log(y') + (1-y)*log(1-y'))`

- Cost Function: `J(w,b) = (1/m) * sum(L(y'[i],y[i]))`
## Equations
Be careful to distinguish np.dot and np.multiply
### Forward propagation
2 equations after vectorization (over `n_x` and over `m`) and broadcasting (of `b`)
1. Linear forward propagation
```
Z[l]=W[l]A[l-1]+b[l]
```
2. Activation forward propagation
```
A[l]=g[l](Z[l])
```
### Backward propagation
1+4 equations after vectorization (over `n_x` and over `m`) and broadcasting (of `b`)

1. Initialization backward propagation (that is the derivative of cost with respect to A[L])
```
dA[L] = (-(Y/A) + ((1-Y)/(1-A)))       # use np.divide
```

and then for f in range(1,L+1)
2. Activation backward propagation
```
# Inputs: dA[l], caches (particularly: Z[l],W[l]) and g'[l]
dZ[l] = dA[l] * g'[l](Z[l])             # /!\ element wise product (*) is np.multiply`
```
3. Linear backward propagation
```
dW[l] = 1/m dZ[l]A[l-1].T
db[l] = 1/m np.sum(dZ[l],axis=1,keepdims=True)
dA[l-1] = W[l].T * dZ[l]
# Outputs: dW[l], db[l], dA[l-1]
```

## How a DL algorithm really works
You'll get different steps in a DL algorithm 👊:
- choose hyperparameters (loss function, #iterations, learning rate, L, n[l], activation functions)
- initialiaze parameters {(W[1],b[1]), ... ,(W[L],b[L])} (small values for `W` and zeros for `b`)
- for #iterations
  - **forward propagation**
    - linear forward of layer 1
    - activation (RELU) forward of layer 1
    - ...
    - linear forward of layer L-1
    - activation (RELU) forward of layer L-1
    - linear forward of layer L
    - *activation (sigmoid) forward of layer L*
  - **cost computation**
  - **backward propagation**
    - *initialization backward propagation of layer L*
    - linear backward propagation of layer L
    - activation backward propagation of layer L-1
    - ...
    - linear backward propagation of layer 1
    - activation backward propagation of layer 0 (useless: STOP!)
  - **update parameters**
- compute accuracy of model on training & test sets  
- predict on unknown set 🤙

Note: there are functions you can compute with small L-layer deep NN (large `L=O(log(n)` and small `n[l]`) that shallower NN would require exponentially more hidden units/neurons to compute (small `L` and very large `n[l]=O(2^n)`)
