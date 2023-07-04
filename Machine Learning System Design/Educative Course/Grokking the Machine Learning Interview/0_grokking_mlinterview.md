Grokking the Machine Learning Interview
- [Introduction](#intro)
- [Practical ML techniques/concept](#pmtc)
- Search ranking
- Feed based system
- Recommendation system
- Self-driving car: image segementation
- Entity linking system
- Ad prediction system

## Introduction
<a id='intro'></a>
what to expect in a machine learning interview:
1. problem-solving/coding: similar to other software engineering coding interviews
2. machine learning understanding: focuses on basic ML concepts such as supervised vs unsupervised learning, reinforcement learning, optimization functions, and the learning process of various ML algorithms
3. career discussion: tends to focus on an individual's resume (previous projects) and behavioral aspects such as the ability to work in teams (conflict resolution) and career motivation
4. machine learning system design discussion: focuses on the inerviewee's ability to solve an end-to-end machine learning problem and consists of open-ended questions
  - eg:
    - build a recommendation system that shows relevant products to users
    - build a visual understanding system for a self-driving car
    - build a search-ranking system
  - In order to answer such questions, the candidates should consider the following approach.
  - ![approach mlsd question](https://github.com/tinghe14/MLE-Learner/blob/bca807154e56615749720b044e15db6dccbd8d11/Machine%20Learning%20System%20Design/Educative%20Course/Grokking%20the%20Machine%20Learning%20Interview/approach%20mlsd%20question.png)
 
Important steps that are mostly common among different ML-based systems:
- exs for ml system questions:
  - build a system that shows relevant ads for search engines
  - extract all persons, locations, and organizations from a given corpus of documents
  - recommend movies to a user on Netflix
1. set up the problem:
   - the question is generally very board. so the fist thing is to ask questions. Asking questions help to close the gap between your understanding of question and the interviewer's expectations from your answer.
   - eg: design a search engine that displays the most relevant results in response to user queries.
     - you could narrow down the problem's scope by asking the following questions:
       - 'is it a general search engine like Google or Bing or a specialized search engine like Amazon's product search?'
       - 'what kind of queries is it expected to answer?'
       - this will help you to precisely define your ml questions, like: 'build a generic search engine that returns relevant results for queries like"Richard Nixon", "Programming Languages" etc'
    - eg: build a system to display a Twitter feed for a user.
      - In this case, you can discuss how the feed is currently displayed and how it can be improved to provide a better experience for the users
        - later chapter, we discuss how the Twitter feed was previously displayed in chronological order, causing users to miss out on relevant tweets. From this discussion, we realized that we want to move towards displaying content in order of relevant instead
        - will get a precise ml problem statement, like 'given a list of tweets, train an ML model that predicts the probability of engagement of tweets and orders them based on that score'
3. understand scale and latency requirements
   - discuss about performance and capacity considerations of the system. It will allow you to clearly understand the scale of the system and its requirements
   - some examples of questions you need to ask:
     - latency requirements:
       - a search engine problem, you can ask 'do we want to return the search result in 100 milliseconds or 500 milliseconds'
       - twitter feed problem, you can ask 'do we want to return the list of relevant tweets in 300 milliseconds or 400 milliseconds'
     - scale of data:
       - a search engine problem, you can ask:
         - 'how many requests per second do we anticipate to handle?'
         - 'how many websites exists that we want to enable through this search engine?'
         - 'if a query has 10 billion matching documentation, how many of these whould be ranked by our model?'
       - twitter feed problem, 'how many tweets would we have to rank according to relevance for a user at a time?'  
   - The answers to these questions will guide you when you come up with the architecture of the system. Knowing that you need to return results quickly will influence the depth and complexity of your models. Having huge amounts of data to process, you will design the system with scalability in mind. Find more on this in the architecture discussion section.
4. define metrics
   - metric for offline testing: test the models' performance during the development phase. We might have generic metrics or sepcific metrics for a certain problem
     - binary classification: AUC, log loss, precision, recall and F1 score
     - search ranking: NDCG
   - metric for online testing:
     - once have selected the best performing models offline, then will use online metrics to test them in the production environment
     - while coming up with online metrics, you may need both component-wise and end-to-end metrics. For example, a search ranking model to display relevant results for search queries. May use a component-wise metric such as NDCG to measure the performance of your model online. Also need an end-to-end metric to look at how the system (search engine) is performing with your new model plugged in. A common end-to-end metric for this scenario: user's engagement and retention rate
5. architecture discussion
   - think about the components of the system and how the data will flow through those components
   - eg: ![simplified version of architectural diagram](https://github.com/tinghe14/MLE-Learner/blob/c6bf1750537c8c214b62cd07a42c785491e1e90a/Machine%20Learning%20System%20Design/Educative%20Course/Grokking%20the%20Machine%20Learning%20Interview/simplified%20version%20of%20architectural%20diagram.png)
   - architecting for scale:
     - eg: asked to build an ML system that displays relevant ads to users. During its problem setup, you ask questions and realize that the number of users and ads in the system is huge and ever-increasing. Thus, need a scalable system that quickly figures out the relevant ads for all users despite the increase in data. Hence, can't jusut build a complex ML model since it takes up a lot of time and resources. The solution is to use the funnel approach,  where each stage will have fewer ads to process.
6. offline model building and evaluation
  - steps:
    - training data generation (hire labelers to human label the data; data collection through a user's interaction with the pre-existing system)-> feature engineering -> model training -> offline evaluation
      - data collection through a user's interaction with the pre-existing system: Q: search engine which shows results for user queries. You can see how people interact with these results to gather training data. If a user clicks a result, you can count it as a positive training example. Similarity, an impression can cound a negative example
      - feature engineering: can start this process by explicitly pinpointing the actors involved in the given task. Q: make movie recommendations to a Netflix user, ![actors](https://github.com/tinghe14/MLE-Learner/blob/a9ab821bede0f34c6929b2272cd8de48994b291a/Machine%20Learning%20System%20Design/Educative%20Course/Grokking%20the%20Machine%20Learning%20Interview/actors.png). In order to make features, you would individually insepct these actors and explore their relationships too.
      - model training: if use the funnel approach, may select simpler models for the top of the funnel where data size is huge and more complex NN or tree based models for successive parts of the funnel. Also have the option of utilizing pre-trained SOTA models to leverage the power of transfer learning
      - offline evaluation: With careful consideration, divide the data into training and validation sets. Use the training data set during the model training part, where you come up with various modeling options (different models, hyperparameters, feature sets, etc.). Afterwards, we evaluate these models, offline, on the validation dataset.
7. online model execution and evaluation
  - Now that you have selected the top-performing models, you will test them in an online environment
  - Depending on the type of problem, you may use both component level and end-to-end metrics. As mentioned before, for the search engine task, the component-wise metric will be NDCG in online testing. However, this alone is not enough. You also need an end-to-end metric as well, like session success rate, to see if the system’s (search engine’s) performance has increased by using your new search ranking ML model.
12. iterative model improvement
  - Your model may perform well during offline testing, but the same increase in performance may not be observed during an online test. Here, you need to think about debugging the model to find out what exactly is causing this behavior.
  - Is a particular component not working correctly? Is the features’ distribution different during training and testing time? For instance, a feature called “user’s top five interest” may show a difference in distribution during training and testing, when plotted. This can help you to identify that the routines used to provide the top five user interests were significantly different during training and testing time.
  - Moreover, after the first version of your model has been built and deployed, you still need to monitor its performance. If the model is not performing as expected, you need to go towards debugging. You may observe a general failure from a decrease in AUC. Or, you may note that the model is failing in particular scenarios. For instance, by analysing the video recording of the self-driving car, you may find out that the image segmentation fails in rushy areas.
  - The problem areas identified during model debugging will guide you in building successive iterations of your model.

## Practical ML techniques/concept
<a id='pmtc'></a>
