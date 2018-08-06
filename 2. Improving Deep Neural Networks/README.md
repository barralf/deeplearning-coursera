## Train/Dev/Test sets :
Be sure all 3 have same distribution.
1. Build a model upon `training set`
2. Then try to optimize hyperparameters on `dev set` as much as possible
3. Then once model is ready, evaluate the `testing set`.

The trend on the ratio of splitting the models:
- if dataset size is 100 to 1000000 ==> 60/20/20
- if dataset size is 1000000 to INF ==> 98/1/1 or 99.5/0.4/0.1

Bias/variance trade-off:
- overfitting: high bias (training set). What to do?
    - make NN bigger
    - try different model
    - try to run longer
    - try different optimization algorithms
- underfitting: high variance (dev set). What to do?
    - use more dataset
    - try regularization
    - use dropout
    - try different model

Ideal bias/variance ratio: **0.5%/1%**

## Initialization summary
The weights `W[l]` should be initialized randomly to **break symmetry**.

The biases `b[l]` can be initialized to zeros. It is ok because symmetry is still broken so long as `W[l]` is initialized randomly.

Different initializations lead to different results

Random initialization is used to break symmetry and make sure different hidden units can learn different things

Don't initialize values too large

He initialization works well for networks with ReLU activations. It initializes the weights to random values scaled according to a paper by He et al., 2015.

## Regularization summary
To be used when overfitting to help reduce the variance. It drives your weights `W[l]_i` to lower values.
1. L2 Regularization
2. Dropout
3. Data augmentation
4. Early stopping

#### 1. L2 Regularization
Formula:

Observations:
`lambda` is the regularization parameter. An hyperparameter that you can tune using a dev set.
L2 regularization makes your decision boundary smoother. If `lambda` is too large, it is also possible to "oversmooth", resulting in a model with lower variance but high bias.

What is L2-regularization actually doing?:
- L2-regularization relies on the assumption that a model with small weights is simpler than a model with large weights. Thus, by penalizing the square values of the weights in the cost function you drive all the weights to smaller values. It becomes too costly for the cost to have large weights! This leads to a smoother model in which the output changes more slowly as the input changes.

What you should remember: the implications of L2-regularization on:
- The cost computation: A regularization term is added to the cost
- The backpropagation function:  There are extra terms in the gradients with respect to weight matrices
- Weights end up smaller ("weight decay"): Weights are pushed to smaller values.

#### 2. Dropout
Another regularization technique. Eliminating some neurons based on a probability.

To be used only during training. NOT to be used during testing.

Apply dropout both during forward and backward propagation.
During training time, divide each dropout layer by keep_prob to keep the same expected value for the activations. For example, if keep_prob is 0.5, then we will on average shut down half the nodes, so the output will be scaled by 0.5 since only the remaining half are contributing to the solution. Dividing by 0.5 is equivalent to multiplying by 2. Hence, the output now has the same expected value.

#### 3. Data augmentation
Image recogn: Flip all your pictures. Randomly position/crop, rotate them.
OCR: random rotations or distorsions.

#### 4. Early stopping
Advantage: no hyperparameter
L2 regularization is better though.


## Normalizing
Standard Normalization with mean and variance. Decreases drastically the computation time.

With unormalized data, cost function can be dee with an inconsistent (elongated) shape.

## Vanishing/exploding gradients
Illustration on a simple NN: a deep NN with large number of layers, same weights and zero biases. Makes activations and gradients explode or vanish

Partial solution: initialization with variance (depending on activation function used)

## Gradient checking with numerical approximations
NOT for training. ONLY for debugging to see if the backward propagation is correct.
```
for i in len(theta):
  d_theta_approx[i] = (J(theta1,...,theta[i] + eps) -  J(theta1,...,theta[i] - eps)) / 2*eps
```

Finally we evaluate this formula
```
(||d_theta_approx - d_theta||) / (||d_theta_approx||+||d_theta||) (|| - Euclidean vector norm)
```
and check (with an epsilon `eps = 10^-7`):
- if it is `< 10^-7` - great, very likely the backward propagation implementation is correct
- if it is around `~10^-5` - can be OK, but need to inspect if there are no particularly big values in `d_theta_approx - d_theta vector`
- if it is `>= 10^-3` - bad, probably there is a bug in backward propagation implementation

Note: Don't forget to add lamda/(2m) * sum(W[l]) to J if you are using L1 or L2 regularization.

---------
# Optimization algorithms

## Batch vs Mini batch gradient descent
Batch gradient descent we run the gradient descent on the whole dataset.

Mini-Batch gradient descent we run the gradient descent on the mini datasets. Works much faster in the large datasets.
Size:
    (mini batch size = m) ==> Batch gradient descent
    (mini batch size = 1) ==> Stochastic gradient descent (SGD)
    (mini batch size = between 1 and m) ==> Mini-batch gradient descent
Batch gradient descent:
- too long per iteration
Stochastic gradient descent:
- too noisy regarding cost minimization (can be reduced by using smaller learning rate)
- won't ever converge (reach the minimum cost)
- lose speedup from vectorization
Mini-batch gradient descent:
- faster learning:
- you have the vectorization advantage
- make progress without waiting to process the entire training set
- doesn't always exactly converge (can oscillate in a very small region, but you can reduce learning rate then)
Guidelines when choosing mini-batch size:
- For small training set (< 2000 examples) - use batch gradient descent.
- Must be a power of 2 (because of the way computer memory is layed out and accessed): 64, 128, 256, 512, 1024, ...
- Make sure that mini-batch fits in CPU/GPU memory.
- Mini-batch size is a hyperparameter.

Steps: random + partition
Notation: `[i]{j}(k)` superscript means i-th layer, j-th minibatch, k-th example.
## Momentum
```
vdW = 0, vdb = 0
on iteration t:
	# can be mini-batch or batch gradient descent
	compute dw, db on current mini-batch                

	vdW = beta * vdW + (1 - beta) * dW
	vdb = beta * vdb + (1 - beta) * db
	W = W - learning_rate * vdW
	b = b - learning_rate * vdb
```

How do you choose  `beta` ?

The larger the momentum  `beta`  is, the smoother the update because the more we take the past gradients into account. But if  `beta`  is too big, it could also smooth out the updates too much.
Common values for  `beta`  range from 0.8 to 0.999. If you don't feel inclined to tune this,  `beta=0.9`  is often a reasonable default.
Tuning the optimal  `beta`  for your model might need trying several values to see what works best in term of reducing the value of the cost function  JJ .
What you should remember:

Momentum takes past gradients into account to smooth out the steps of gradient descent. It can be applied with batch gradient descent, mini-batch gradient descent or stochastic gradient descent.
You have to tune a momentum hyperparameter `beta` and a learning rate `alpha`.

## RMSprop
```
sdW = 0, sdb = 0
on iteration t:
	# can be mini-batch or batch gradient descent
	compute dW, db on current mini-batch

	sdW = (beta * sdW) + (1 - beta) * dW^2  # squaring is element-wise
	sdb = (beta * sdb) + (1 - beta) * db^2  # squaring is element-wise
	W = W - learning_rate * dW / sqrt(sdW + epsilon) # epsilon=10^-8
	b = B - learning_rate * db / sqrt(sdb + epsilon) # epsilon=10^-8

```
## ADAM
```
vdW = 0, vdW = 0
sdW = 0, sdb = 0
on iteration t:
	# can be mini-batch or batch gradient descent
	compute dw, db on current mini-batch                

	vdW = (beta1 * vdW) + (1 - beta1) * dW     # momentum
	vdb = (beta1 * vdb) + (1 - beta1) * db     # momentum

	sdW = (beta2 * sdW) + (1 - beta2) * dW^2   # RMSprop
	sdb = (beta2 * sdb) + (1 - beta2) * db^2   # RMSprop

	vdW = vdW / (1 - beta1^t)      # fixing bias
	vdb = vdb / (1 - beta1^t)      # fixing bias

	sdW = sdW / (1 - beta2^t)      # fixing bias
	sdb = sdb / (1 - beta2^t)      # fixing bias

	W = W - learning_rate * vdW / (sqrt(sdW) + epsilon) # epsilon=10^-8
	b = B - learning_rate * vdb / (sqrt(sdb) + epsilon) # epsilon=10^-8
```

Hyperparameters for Adam:
- `learning_rate`: needed to be tuned. See learning rate decay.
- `beta1`: parameter of the momentum - 0.9 is recommended by default.
- `beta2`: parameter of the RMSprop - 0.999 is recommended by default.
- `epsilon`: 10^-8 is recommended by default.

Advantages of Adam :
- Relatively low memory requirements (though higher than gradient descent and gradient descent with momentum)
- Usually works well even with little tuning of hyperparameters (except `learning_rate` )

## Learning rate decay
slowly reduce learning rate over during iterations so it makes it closer to the optimum because steps will be smaller and more accurate.

Possible learning rate decay methods:
- learning_rate = (1 / (1 + decay_rate * epoch_num)) * learning_rate_0
- learning_rate = (0.95 ^ epoch_num) * learning_rate_0
- learning_rate = (k / sqrt(epoch_num)) * learning_rate_0
- manually

Hyperparameters for learning rate decay:
- `learning_rate_0`
- `decay_rate`

How to compute the cost function faster:
- Try mini-batch gradient descent
- Try better random initialization for the weights
- Try using Adam
- Try tuning the learning rate `learning_rate`


# Hyperparameter tuning, Batch Normalization and Programming Frameworks
## Tuning process
Hyperparameters importance are (as for Andrew Ng):
- Learning rate. (more critical than other hyperparameters)
- Momentum beta.
- Mini-batch size.
- No. of hidden units.
- No. of layers.
- Learning rate decay.
- Regularization lambda.
- Activation functions.
- Adam beta1 & beta2.
Try random values (not grid). Use coarse to fine (zoom on small region).


## Tuning on logarithmic scale
Appropriate scale is logarithmic scale

Babysitting model
Caviar model: different models in parallel if enough computational ressources !!

## Batch Normalization
Normalizing activations.

`W[l]`, `b[l]` become parameters `W[l]`, `gamma[l]`, `beta[l]`
Pseudo code:
```
for t=1...NumberMiniBatches
  compute forward propagation on X{t}
    in each hidden layer, use BN to Z[l]->Z_tilde[l]
  compute backward propagation: dW[l], dbeta[l],dgamma[l]
  update parameters
    W[l]=W[l]-learning_rate*dW[l]
    beta[l]=beta[l]-learning_rate*dbeta[l]
    gamma[l]=gamma[l]-learning_rate*dgamma[l]
```
2 new parameters
## Softmax regression

Generalization of logistic regression to multiclass classification/regression
`C`: number of classes. If `C=2`, softmax regression is actually a simple logistic regression.
`N_y=C`

Loss function: `L(y, y_hat) = - sum(y[j] * log(y_hat[j])) # j = 0 to C-1`

## TensorFlow
Writing and running programs in TensorFlow has the following steps:
1. Create Tensors (variables) that are not yet executed/evaluated.
2. Write operations between those Tensors.
3. Initialize your Tensors.
4. Create a Session.
5. Run the Session. This will run the operations you'd written above.

# Random clarifications
## difference epoch_num vs iterations:

Epoch: number of times the algorithm sees the entire data set. So, each time the algorithm has seen all samples in the dataset, an epoch has completed.

Iteration: the number of times a batch of data passed through the NN

Example of stackoverflow:
Say you have a dataset of 10 examples. You have a batch size of 2, and you've specified you want the algorithm to run for 3 epochs.
Therefore, in each epoch, you have 5 batches (10/2 = 5). Each batch gets passed through the algorithm, therefore you have 5 iterations per epoch. Since you've specified 3 epochs, you have a total of 15 iterations (5*3 = 15) for training.
