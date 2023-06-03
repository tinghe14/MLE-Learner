Course Content
- Week 1: Introduction to Deep Learning
- Week 2: Neural Networks Basic
- Week 3: Shallow Neural Networks
- Week 4: Deep Neural Networks

# W1: Introduction to Deep Learning
none

# W2: Neural Networks Basic
Set up a machine learning problem with a neural network mindset and use vectorization to speed up your models.

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
  - real loss function of logistic reg which is convex: -ylogy_hat - (1-y)log(1-y_hat)
    - some intuitions on why this loss function makes sense
    - we want to make it smaller
    - if y=1, Loss = -logy_hat -> want y_hat to be large because y_hat is you know the sigmoid function, it can never be bigger than one
    - if y=0, loss = -log(1 - y_hat)
- cost function: works on entire dataset, so is going to be the average of sum of loss function apply to each training sample at a time -> 1/m * sum of loss function

### Logistic regression as neural network
gradient descent: to learn and train your parameter
- initialize w and b to some initial value (for logistic regression any initlization can work, usually you initialize the values of 0; because the function is convext no matter what the initalization method choose, you should get to the same point or roughly the same point)
- gradient descent does is it starts at that initial point and then takes a step in the steepest downhill direction as quickly down as possible, and repeat the process until the algorithm converage

derivatives

 
### python and vectorization
### programming assignments
