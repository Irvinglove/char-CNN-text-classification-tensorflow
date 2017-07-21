# char-CNN-text-classification-tensorflow

## Here is an article explaining this code ##

## Reqirement ##
- Python 2.7.6
- Numpy 1.13.1
- TensorFlow 1.2.1

## Running ##
python training.py

## Models ##
charCNN.py : 9-layer large convolutional neural network based on raw character.

## Dataset ##
If dataset is not found under datasets_dir, it will be downloaded automatically. 
The feeding method is used now to get data into TF model.

-- ag: [AG](http://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html) is a collection of more than 1 million 
news articles. News articles have been gathered from more than 2000 news sources by ComeToMyHead in more than 1 year of 
activity. ComeToMyHead is an academic news search engine which has been running since July, 2004.
