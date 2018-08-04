# Initialization summary
-The weights `W[l]` should be initialized randomly to break symmetry

The biases `b[l]` can be initialized to zeros. It is ok because symmetry is still broken as long as `W[l]` is initialized randomly.

Different initializations lead to different results

Random initialization is used to break symmetry and make sure different hidden units can learn different things

Don't initialize values too large

He initialization works well for networks with ReLU activations.

# Regularization summary
To be used when overfitting to help reduce the variance. It makes `W[l]_i` smaller.
## 1. L2 Regularization

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

## 2. Dropout
Another regularization technique. Eliminating some neurons based on a probability.

To be used only during training. NOT to be used during testing.

Apply dropout both during forward and backward propagation.
During training time, divide each dropout layer by keep_prob to keep the same expected value for the activations. For example, if keep_prob is 0.5, then we will on average shut down half the nodes, so the output will be scaled by 0.5 since only the remaining half are contributing to the solution. Dividing by 0.5 is equivalent to multiplying by 2. Hence, the output now has the same expected value.

## 3. Data augmentation
Image recogn: Flip all your pictures. Randomly position/crop, rotate them.
OCR: random rotations or distorsions.

## 4. Early stopping
Advantage: no hyperparameter
L2 regularization is better though.


# Normalizing
Standard Normalization with mean and variance. Decreases drastically the computation time.

With unormalized data, cost function can be dee with an inconsistent (elongated) shape.

# Vanishing/exploding gradients
Illustration on a simple NN: a deep NN with large number of layers, same weights and zero biases. Makes activations and gradients explode or vanish

Partial solution: initialization with variance (depending on activation function used)
