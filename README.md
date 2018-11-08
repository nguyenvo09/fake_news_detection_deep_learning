# Fake News Detection With Deep Learning
This repository is for Fake News Detection using Deep Learning models

## Motivation
Fake news is widely spread across social network. Therefore, there is a huge demand to debunk fake news. There are many attempts to detect fake news but limited work is about using Deep Learning models. In this project, we aim to build state-of-the-art deep learning models to detect fake news based on the content of article itself. 
![alt text](https://github.com/nguyenvo09/fake_news_detection_deep_learning/blob/master/images/wordcloud.png)


### Dataset
We get the ground truth data from https://www.kaggle.com/arminehn/rumor-citation/data#. We only use Snopes URLs since the labels of each news were clearly presented. Only "true" or "false" labels were kept. We randomly select 281 true news and 281 false news and crawl the Snope website for additional content. 

### Phases of the project
* Building machine learning models to detect fake news 
* Using multiple data sources to detect fake news. 

### Current progress
We already built many traditional machine learning models as baselines. We also already implemented a deep learning model called Bi-directional GRU with Attention mechanism which was originally proposed by Yang el al., [1]. We implemented this model for fake news detection domain. Some visualizations of attention weights learned by this model is shown in https://github.com/nguyenvo09/fake_news_detection_deep_learning/blob/master/biGRU_attention.ipynb (Note: Github does not show the CSS of our visualizations. It is better to view visualizations in your local machines). We also use the Attention layer code implemented by @ilivans. Thanks. We implement all Deep Learning models in Tensorflow 1.4

![alt text](https://github.com/nguyenvo09/fake_news_detection_deep_learning/blob/master/images/visualize1.PNG)
and 
![alt text](https://github.com/nguyenvo09/fake_news_detection_deep_learning/blob/master/images/visualize2.PNG)

### Future work
* Trying other deep learning models such as Auto-Encoders, GAN, CNN

### References
[1] Hierarchical Attention Networks for Document Classification, Zichao Yang, Diyi Yang, Chris Dyer, Xiaodong He, Alex Smola, Eduard Hovy, 2016 NAACL 



