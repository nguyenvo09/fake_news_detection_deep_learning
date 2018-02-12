# fake_news_detection_deep_learning
This repository is for Fake News Detection using Deep Learning models

This project consists of two main phases: (1) building machine learning models to detect fake news and (2) using multiple data sources to detect fake news. 

We get the ground truth data from https://www.kaggle.com/arminehn/rumor-citation/data#. We only use Snopes URLs since the labels of each news were clearly presented. Only "true" or "false" labels were kept. We randomly select 281 true news and 281 false news and crawl the Snope website for additional content. 
