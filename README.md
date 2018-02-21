# fake_news_detection_deep_learning
This repository is for Fake News Detection using Deep Learning models

## Motivation
Fake news is widely spread across social network. Therefore, there is a huge demand to debunk fake news. There are many attempts to detect fake news but limited work is about using Deep Learning models. In this project, we aim to build state-of-the-art deep learning models to detect fake news based on the content of article itself. 
### Phases of the project
* building machine learning models to detect fake news 
* using multiple data sources to detect fake news. 

We get the ground truth data from https://www.kaggle.com/arminehn/rumor-citation/data#. We only use Snopes URLs since the labels of each news were clearly presented. Only "true" or "false" labels were kept. We randomly select 281 true news and 281 false news and crawl the Snope website for additional content. 
