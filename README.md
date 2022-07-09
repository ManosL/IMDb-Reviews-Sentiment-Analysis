# IMDb-Reviews-Sentiment-Analysis

## Overview

In this project I try to tackle the problem of Sentiment Analysis using various
Machine Learning and Deep Learning approaches. The dataset that was used was the
[IMDb Movie Reviews Dataset](https://paperswithcode.com/dataset/imdb-movie-reviews) which contains 
positive and negative reviews because it contained 25000 train instances and 25000 test ones, thus,
it was feasible to train the models in our local machine, and also there was the [Sentiment Analysis Leaderboard](https://paperswithcode.com/sota/sentiment-analysis-on-imdb), hence, I could see how good our approaches are in comparison to state of the art models. The proposed approaches are the following:

1. Machine Learning Approach using Preprocessing and Feature Extraction techniques and different classifiers(SVM and Naive Bayes specifically).
2. LSTM with trained embeddings.
3. Transfer Learning with BERT.

Out of all the approaches and after some hyperparameter tuning, I determined that the best approach was the last one because it yielded accuracy equal to 89.5%. In comparison, with the state of the art models as of 09/07/2022 this approaches is placed 27th according to the aforementioned Leaderboard.

## Manual

In order to run the experiments you should follow these steps:

1. Create and activate a virtual environment and then run the command
pip3 install -r requirements.txt in order to install the necessary
libraries in order to run the code

2. Unzip the imdb.zip. In this file there is contained the
IMDb reviews dataset.

3. For Machine Learning Experiments run the ./code/ml_approach.py
using Python3.

4. For LSTM experiments you can either run the ./code/lstm_experiments.sh
script that runs the Grid Search I present on this report or the
./code/lstm_approach.py script in order to run a single experiment. You can
add the parameter -h in order to get the usage of that file.

5. For BERT experiments you can either run the ./code/bert_experiments.sh
script that runs the Grid Search I present on this report or the
./code/bert_approach.py script in order to run a single experiment. You can
add the parameter -h in order to get the usage of that file.

6. In the ./code/{bert|lstm}_logs directories contain the output of each experiment.

## Further Work

In the future, I can use another pre-trained models, bigger models or ensembles, but I believe that there is still room for improvement using those models in terms of handling the data, because in essence I currently just feed the data as the are(except for the basic preprocessing) and instead I need to find a more sophisticated approach that exploits its structure and address their weaknesses. Some thoughts are the following:

1. Try bidirectional LSTM. This is expected to yield better performance because
it does not only consider for each word the words before but also the words
after. Then you need to take the "leftmost" and "rightmost" outputs and pass
them to Linear layer.

2. You should find some way to mitigate the following case. There is the case
that the review will be like:

"Jack Nicolsson rocks! Movie sucks..." or "Jack Nicolsson rocks, but movie 
sucks" or "the structure is good, but product doesn't work"

This means that the classifier will be confused without proper algorithm. This
includes the existing ones. Cutting tokens won't work because I will lose 
information about the part that I care. Using TF-IDF just neutralizes if I have
a positive and negative part in the sentence.

You can take each sentence/phrase to predict it and use ensembles to draw 
the final prediction or just pass them to Logistic Regression classifier,
for instance.

3. I should be careful with stopword removal because "not" is also a
stopwords. No need to say more.

Finally, I can evaluate how my proposed approaches perform on different
kind of data, like product reviews, tweets etc.