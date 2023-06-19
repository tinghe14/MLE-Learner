Course Content
- Week 1: Introduction to Deep Learning
- Week 2: Neural Networks Basic
- Week 3: Shallow Neural Networks
- Week 4: Deep Neural Networks

# W1: Introduction to Deep Learning
none

# W2: Neural Networks Basic
Set up a machine learning problem with a neural network mindset and use vectorization to speed up your models.
### Logistic regression as neural network

To make implementation easier, it would be convenient to stack X(features) in columns (so do for the Y(output)) feature matrix, X: R[n_x x m_x], output matrix, Y: R[1 x m]
- n_x rows: n_x features
- m_x columns: m_x training samples

logistic regression: y_hat = P(y=1 | x), when training the logistic regression, the job is to try to learn parameters w and b so that y_hat becomes a good estimate of the chance of Y being equal to one
- x: R[n_x]
- parameters: w:R[n_x], b:R
- output y_hat = $`\textcolor{red}{\text{sigmoid}}`$(w_transpose x x + b)
  - inner function can be much bigger than one or it can even be negative, which doesn't make sense for probability that you want to output to be zero to one. 
  - so use sigmoid function applied to this quantity
 ![sigmoid](https://github.com/tinghe14/MLE-Learner/blob/34728ab2815cafe4cfa3f7f92d49c2d71aaf7efd/Deep%20Learning/Deep%20Learning%20Specialization%20from%20deeplearning.ai/sigmoid.png)
 
 logistic regression cost function
 - notation: i superscript means the ith example
 - loss function: is a function to measure how good output y_hat is when the true label is y
  - might seems nature to use square error, but in logistic regression people don't usually do this. Beacuse when you come to learn the parameters, it is an optmization problem. it is non-convex which means when applying gradient descent, you may end up with many local optima, may not find a global optimum
    - negative log likelihood cost: $`\textcolor{red}{\text{real loss function of logistic reg which is convex: -ylogyhat - (1-y)log(1-yhat)}}`$; more general form/; [cross entroy](https://en.wikipedia.org/wiki/Cross_entropy#Cross-entropy_error_function_and_logistic_regression)
    - some intuitions on why this loss function makes sense
    - we want to make it smaller
    - if y=1, Loss = -logy_hat -> want y_hat to be large because y_hat is you know the sigmoid function, it can never be bigger than one
    - if y=0, loss = -log(1 - y_hat)
- cost function: works on entire dataset, so is going to be the average of sum of loss function apply to each training sample at a time -> 1/m * sum of loss function

gradient descent: to learn and train your parameter
- initialize w and b to some initial value (for logistic regression any initlization can work, usually you initialize the values of 0; because the function is convext no matter what the initalization method choose, you should get to the same point or roughly the same point)
- gradient descent does is it starts at that initial point and then takes a step in the steepest downhill direction as quickly down as possible, and repeat the process until the algorithm converage

derivates with a computation graph
- forward: left to right calculation to compute the cost function you might want to optmize
- back propagation: right to left calculation to compute derivatives
- chain rule

gradient descent on m examples
- overall gradient: the average of derivates respect to w_1 of the individual loss terms
![0](https://github.com/tinghe14/MLE-Learner/blob/9a8afa2d5e5fbd31e180d63c7b45ee4a4cfb6162/Deep%20Learning/Deep%20Learning%20Specialization%20from%20deeplearning.ai/0_gradient%20descent%20on%20m%20example%200.png)
![1](https://github.com/tinghe14/MLE-Learner/blob/9a8afa2d5e5fbd31e180d63c7b45ee4a4cfb6162/Deep%20Learning/Deep%20Learning%20Specialization%20from%20deeplearning.ai/0_gradient%20descent%20on%20m%20example%201.png)
  - two weakness with this calculation:
    - to implement logistic regression this way, you end up writing two for loops
      - the first for loop is for loop over the m training examples
      - the second for loop is for a for loop over n features over here
    - without using explicit for loops is important and will help you to scale to much bigger datasets: vectorization
 
### python and vectorization
vectorization:
- ![0_vectorization vs non vectorization examples](https://github.com/tinghe14/MLE-Learner/blob/9a6171e97008883fdca58b39efb370f7d75c887d/Deep%20Learning/Deep%20Learning%20Specialization%20from%20deeplearning.ai/0_vectorization%20vs%20non%20vectorization%20examples.png)
  - CPU and GPU both have parallization instructions. This basically means is that, if you use built-in functions such as np.function or other functions that don't require you explicity implementing a for loop. It enables python Pi to take much better advantage of parallelism to do your computations much faster

vectorizing logistic regression's gradient output
- without any for loop
  - ![0 without any for loop](https://github.com/tinghe14/MLE-Learner/blob/30779c02647daafe2ef2fe32c8e75d781419f2e8/Deep%20Learning/Deep%20Learning%20Specialization%20from%20deeplearning.ai/0%20without%20any%20for%20loop.png)

multi iterations of gradient descent update for logistic regression
- there is an outermost for loop for multi iterations, don't think you can get rid of that
- ![0 gradient descent vectorized logistic regression](https://github.com/tinghe14/MLE-Learner/blob/cc0d0acb27f5d4ab83fab5675784c22d92125290/Deep%20Learning/Deep%20Learning%20Specialization%20from%20deeplearning.ai/0%20gradient%20descent%20vectorized%20logistic%20regression.png)

a note on python/numpy vectors
- np.random.randn(5): generate a ranked 1 array with output of one squared bracket only, it is not a row vector or column vector, can have some subtle errors which are hard to debug, such as a = np.random.rand(5); b=a.T; but np.dot(a, b) will not be a matrix, it is a single number which is not expected as we want
- np.random.randn((5,1)): generate a row vector with output of tow squared brackets, recommend to use this one
- if not sure, recommend to use assert or reshape, they are inexpensive to use: assert(a.shape == (5,1)) or a = a.reshape((5,1))

Quiz:
- Suppose x is a (8,1) array. Which of the following is a valid reshape? A. x.reshape(-1, 3); B. x.reshape(2,2,2) -> B: this requires that 8 = n*3 where n is an unknown size to be determined. N need to be a integer

Practice Programming Assignment:
- https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html: 

### programming assignments
