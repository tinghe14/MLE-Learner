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
performance and capacity considerations
- as we work on a machine learning based system, our goal is generally to improve our metrics (engagement rate, etc). While ensuring that we meet the capacity and performance requirement
- ml algorithms have 3 different types of complexities
  - training complexity: time taken by it to train the model for a given task
  - evaluation complexity: time taken by it to evaluate the input at testing time
  - sample complexity: total number of training samples required to learn a target function successfully (Sample complexity changes if the model capacity changes. For example, for a deep neural network, the number of training examples has to be considerably larger than decision trees and linear regression)
- ![comparision of training and evaluation complexities](https://github.com/tinghe14/MLE-Learner/blob/aa7a852891ec281caf9dd48240664b0374898c7e/Machine%20Learning%20System%20Design/Educative%20Course/Grokking%20the%20Machine%20Learning%20Interview/comparision%20of%20training%20and%20evaluation%20complexities.png)
- techniques to boost compacity:
  - distributed system: we will distribute the load of a single query among multiple shards, e.g., we can divide the load among 1000 machines and can still execute our fast model on 100 million documents in 100ms (100s/1000)
  - funnel-based approach: start with a relatively fast model when you have the most number of documents;  In every later stage, we continue to increase the complexity (i.e. more optimized model in prediction) and execution time but now the model needs to run on a reduce number of documents e.g. our first stage could use a linear model and final stage can use a deep neural network.
    - In ML systems like search ranking, recommendation, and ad prediction, the layered/funnel approach to modeling is the right way to solve for scale and relevance while keeping performance high and capacity in check

training data collection strategies
- collection techniques
  - user's interaction with pre-existing system (online):
    - the early version is a rule-based system. With the rule-based system in place, you build an ML system for the task (which is then iteratively improved). So when you build the ML system, you can utilize the user's interaction with the prevuious system to generate training data for model training
    - eg: ![Netflix recommendation system](https://github.com/tinghe14/MLE-Learner/blob/78514e45d3b676c572280241f3d837ccf6b49feb/Machine%20Learning%20System%20Design/Educative%20Course/Grokking%20the%20Machine%20Learning%20Interview/Netflix%20recommendation%20system.png)
      - The early version for movie recommendation might be popularity-based, localization-based, rating-based, hand created model or ML-based. The important point here is that we can get training data from the user’s interaction with this system. If a user likes/watches a movie recommendation, it will count as a positive training example, but if a user dislikes/ignores a movie recommendation, it will be seen as a negative training example.
  - human labelers (offline):
    - In other cases, the user of the system would not be able to generate training data. Here, you will utilize labelers to generate good quality training data.
    - eg: Assume that you are asked to perform image segmentation of the surroundings of a self-driving vehicle. The self-driving car will have a camera to take pictures of its surroundings. You will be training a model that will segment these pictures/images into various objects such as the road, pedestrians, building, sidewalk, signal and so on. For this, you will need to provide accurately segmented images of the self-driving car’s surroundings as training data to the model.Here, the consumer of the system (the person sitting in the self-driving car) can’t generate training data for us. They are not interacting with the system in a way that would give segmentation labels for the images captured by the camera.
    - crowdsourcing: collect training data for relatively simpler tasks, eg: email spam detection system
    - specialized labelers: trained labelers will use software, such as Label box, to mark the boundaries of different objects in the driving images
      - targeted data gathering: Offline training data collection is expensive. So, you need to identify what kind of training data is more important and then target its collection more. To do this, you should see where the system is failing, i.e., areas where the system is unable to predict accurately. Your focus should be to collect training data for these areas.
Continuing with the autonomous vehicle example, you would see where your segmentation system is failing and collect data for those scenarios. For instance, you may find that it performs poorly for night time images and where multiple pedestrians are present. Therefore, you will focus more on gathering and labeling night time images and those with multiple pedestrians.
    - open-source datasets: For instance, “BDD100K: A Large-scale Diverse Driving Video Database” is an example of an open-source dataset that can be used as training data for this segmentation task. It contains labeled segmented data for driving images.
  - additional creative collection techniques:
    - build the product in a way that it collects data from user:  We can tweak the functionality of our product in a way that it starts generating training data for our model. Let’s consider an example where people go to explore their interests on Pinterest. You want to show a personalized selection of pins to the new users to kickstart their experience. This requires data that would give you a semantic understanding of the user and the pin. This can be done by tweaking the system in the following way:Ask users to name the board (collection) to which they save each pin. The name of the board will help to categorize the pins according to their content.Ask new users to choose their interests in terms of the board names specified by existing users.
The first step will help you to build content profiles. Whereas, the second step will help you build user profiles. The model can utilize these to show pins that would interest the user, personalizing the experience.
    - eg: ![pinterest example](https://github.com/tinghe14/MLE-Learner/blob/45370ff843d861fbdbed6073c6fdf1f7100410cb/Machine%20Learning%20System%20Design/Educative%20Course/Grokking%20the%20Machine%20Learning%20Interview/pinterest%20example.png)
    - creative manuall expansion: he enhanced training data will enable us to build a more robust logo detector model, which will be able to identify logos of all sizes at different positions and from various kinds of images.
      - eg:![enhaced log system](https://github.com/tinghe14/MLE-Learner/blob/88305588c9fba49c8a1590428e22feb35c56a2f2/Machine%20Learning%20System%20Design/Educative%20Course/Grokking%20the%20Machine%20Learning%20Interview/enhanced%20logo%20system.png)
    - data expansion using GANs: There are a lot of training images of sunny weather conditions but less of rainy weather. A model trained on this training data may not work well for images with rainy conditions. Here, we can utilize GANs to convert images with sunny weather conditions to rainy weather conditions. This will increase our training data for rainy conditions, resulting in a more robust model
- train, test & validation splits:
  - training data: helps in training the ML model (fit model parameters)
    - ![model hyperparameter vs model parameter](https://github.com/tinghe14/MLE-Learner/blob/2ec47957af4e9093ca5392acb8f27da9194168dd/Machine%20Learning%20System%20Design/Educative%20Course/Grokking%20the%20Machine%20Learning%20Interview/model%20hyperparameter%20vs%20model%20parameter.png)
  - validation data: After training the model, we need to tune its hyperparameters (try different model configurations). This process requires testing the model’s performance on various hyperparameter combinations to select the best one.
  - test data: Now that we have trained and tuned the model, the final step is to test its performance on data that it has not seen before. In other words, we will be testing the model’s generalization ability.
  - points to consider during spliting:
    - the size of each split will depend on your particular scenarios: The training data will generally be the largest portion, especially if you are training a model like a deep neural network that requires a lot of training data. Common ratios used for training, validation and test splits are 60%, 20%, 20% or 70%, 15%, 15% respectively
    - While splitting training data, you need to ensure that you are capturing all kinds of patterns in each split. For example, if you are building a movie recommendation system like Netflix, your training data would consist of users’ interactions with the movies on the platform. After analysing the data, you may conclude that, generally, the user’s interaction patterns differ throughout the week. Different genres and movie lengths are preferred on different days. Hence, you will use the interaction with recommendations throughout a week to capture all patterns in each split
    - you will train the model on data from one time interval and validate/test it on the data from its succeeding time interval, as shown in the diagram below. This will provide a more accurate picture of how our model will perform
- quantity of training data
  - As a general guideline, the quantity of the training data required depends on the modeling technique you are using
  - Gathering a large amount of training data requires time and effort. Moreover, the model training time and cost also increase as we increase the quantity of training data. To see the optimal amount of training data, you can plot the model’s performance against the number of training data samples, as shown below. After a certain quantity of training data, you can observe that there isn’t any gain in the model’s performance
- training data filtering
  - It is essential to filter your training data since the model is going to learn directly from it. Any discrepancies in the data will affect the learning of the model
  - cleaning up data: General guidelines are available for data cleaning such as handling missing data, outliers, duplicates and dropping out irrelevant features. Apart from this, you need to analyze the data with regards to the given task to identify patterns that are not useful. For instance, consider that you are building a search engine’s result ranking system. Cases where the user clicks on a search result are considered positive examples. In comparison, those with just an impression are considered negative examples. You might see that the training data consist of a lot of bot traffic apart from the real user interactions. Bot traffic would just contain impressions and no clicks. This would introduce a lot of wrong negative examples. So we need to exclude them from the training data so that our model doesn’t learn from wrong examples.
  - removing bias: The pre-existing recommender is showing recommendations based on popularity. As such, the popular movies always appear first and new movies, although they are good, always appear later on as they have less user engagement. Ideally, the user should go over the whole recommendation list so that he/she can discover the good, new movies as well. This would allow us to classify them as positive training examples so that the model can learn to put them on top, once re-trained. However, due to the user’s time constraints, he/she would only interact with the topmost recommendations resulting in the generation of biased training data. The model hence trained, will continue considering the previous top recommendation to be the top recommendation this time too. Hence, the “rich getting richer” cycle will continue. In order to break this cycle, we need to employ an exploration technique that explores the whole content pool (all movies available on Netflix). Therefore, we show “randomized” recommendations instead of “popular first” for a small portion of traffic for gathering training data. The users’ engagement with the randomized recommendations provides us with unbiased training data. This data really helps in removing the current positional and engagement bias of the system.
    - ![remove bias in recommendation system](https://github.com/tinghe14/MLE-Learner/blob/8d789cda8fcc1eef6f5310aedd9da090795b11e2/Machine%20Learning%20System%20Design/Educative%20Course/Grokking%20the%20Machine%20Learning%20Interview/remove%20bias%20in%20recommendation%20system.png)
  - bootstrapping new items:Sometimes we are dealing with systems in which new items are added frequently. The new items may not garner a lot of attention, so we need to boost them to increase their visibility. For example, in the movie recommendation system, new movies face the cold start problem. We can boost new movies by recommending them based on their similarity with the user’s already watched movies, instead of waiting for the new movies to catch the attention of a user by themselves. Similarly, we may be building a system to display ads, and the new ads face the cold start problem. We can boost them by increasing their relevance scores a little, thereby artificially increasing their chance of being viewed by a person.

online experimentation
embeddings
transfer learning
model debugging and testing
