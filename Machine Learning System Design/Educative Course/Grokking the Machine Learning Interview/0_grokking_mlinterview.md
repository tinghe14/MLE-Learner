Grokking the Machine Learning Interview
- [Introduction](#intro)
- Practical ML techniques/concept
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
5. define metrics
6. architecture discussion
7. offline model building and evaluation
8. online model execution and evaluation
9. iterative model improvement
