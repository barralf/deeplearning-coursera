

### Ship it quickly to deep dive progressively (not too early)
ML is a highly iterative process. If you train a basic model and carry out error analysis, it will help point you in more promising directions.

Orthogonalization: one control does a specific task and does not affect the other controls.

## Satisfacing and optimizing metrics
It's hard to get one single evaluation metric.
In general case:
```
MAXIMIZE one metric        # optimizing the target metric
SUBJECT TO n-1 metrics     # satisfacing all the other metrics
```

Note: F1 score combines Precision and Recall `F1=2/(1/P+1/R)`
## Train/Dev/Test
- Dev/Test must have the same distribution. But it is ok if training set comes from a slightly different distribution
- Size:
  - 60/20/20 (less than 100,000 examples)
  - 98/1/1 (more than 1,000,000 examples)

- Bayes optimal error
- Human level performance (can be a proxy of Bayes optimal error)


## Error analysis

Error analysis: process of manually examining mistakes that your algorithm is making, aka the "errors due to". It can give you game changing insights in what to do next and make much better prioritization on possible promising (or not) actions to take. (ex: work on big cats images, blurry images, dogs images)

#### Correcting one category of error
You can reduce a category of error by adding new data associated (for example: incorrectly labeled data or foggy pictures).
⚠️⚠️ Be aware it is important that your `dev set` and `test set` have **the closest possible distribution to “real”-data**. It is also important for the training set to contain enough “real”-data to avoid having a data-mismatch problem.

## Bias & variance analysis
train-dev set: a random subset of the training set (so it has the same distribution)

- `Avoidable bias = training error - human level error`  --> High bias problem
- `Variance = train dev error - training Error`  --> High variance problem
- `Data mismatch = dev error - train dev error`  --> Data mismatch problem (more similar training data with artificial data synthesis for example or more dev/test data)

## Transfer learning
Task A: your are transferring from. **Pretraining**.
Task B: your are transferring to with **Fine Tuning**.

Transfer learning makes sense in following cases:
- Task A and B have the same input X (e.g. image, audio).
- You have a lot of data for the task A than for task B
- Low level features from task A could be helpful for learning task B.

## Multi-task learning
Multi-task learning makes sense when:
- Training on a set of tasks that could benefit from having shared lower-level features.
- Usually, amount of data you have for each task is quite similar.
- Can train a big enough network to do well on all the tasks. (the performance of the multi-task learning will be better compared to splitting the tasks.

## End-to-end deep learning
Using **end-to-end** deep learning
  - Pros of end-to-end deep learning:
    - Let the data speak. By having a pure machine learning approach, your NN learning input from X to Y may be more able to capture whatever statistics are in the data, rather than being forced to reflect human preconceptions.
    - Less hand-designing of components needed.
  - Cons of end-to-end deep learning:
    - May need a large amount of data.
    - Excludes potentially useful hand-design components (it helps more on the smaller dataset).

#### Applying end-to-end deep learning:
  - **Large amount of data available**. (Key question to ask: Do you have sufficient data to learn a function of the complexity needed to map x to y?()
  - Use DL to learn some individual components.
  - **Decide clearning the mapping (X,Y)**. When applying supervised learning you should carefully choose what types of X to Y mappings you want to learn depending on what task you can get data for.
