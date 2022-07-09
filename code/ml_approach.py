from copy import deepcopy
import pickle
import re

from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import gensim
import gensim.downloader

from utils import read_imdb_data, convert_ratings_to_sentiments



DATASET_PATH = '../IMDB/aclImdb'

# This function contains the preprocessing that should be done in
# all our data regardless if they will be on train or test set
def data_preprocessing(instances):
    processed_instances = pd.Series(instances)

    # Lowercase each review
    processed_instances = processed_instances.apply(lambda x: x.lower())

    # Replace sequent whitespaces with only one
    processed_instances = processed_instances.apply(lambda x: ' '.join(x.split()))

    # Tokenize each review and remove the stopwords
    stopwords_eng = set(stopwords.words('english'))

    processed_instances = processed_instances.apply(lambda x: word_tokenize(x))
    processed_instances = processed_instances.apply(lambda x: [w for w in x if w not in stopwords_eng])

    # Stemming
    stemmer = PorterStemmer()
    processed_instances = processed_instances.apply(lambda x: [stemmer.stem(w) for w in x])

    # Returning the reviews to their prior "string" form
    processed_instances = processed_instances.apply(lambda x: ' '.join(x))

    return processed_instances



def train_and_get_word2vec_model(instances, vec_size=300):
    # Training word2vec model
    model_w2v = gensim.models.Word2Vec(
                instances,
                vector_size=vec_size, # desired no. of features/independent variables
                window=5, # context window size
                min_count=2,
                sg = 1, # 1 for skip-gram model
                hs = 0,
                negative = 10, # for negative sampling
                workers= 4, # no.of cores
                seed = 34)

    return model_w2v



def extract_word2vec_instances(model_w2v, instances):
    new_instances = []

    for instance in instances:
        tokens = word_tokenize(instance)
        
        new_instance = np.zeros(model_w2v.vector_size)
        words        = 0
        
        for token in tokens:
            if token in model_w2v.wv:
                new_instance += model_w2v.wv[token]
                words += 1
        
        if words == 0:
            words = 1

        new_instances.append(list(new_instance / words))

    return np.array(new_instances)



def svm_preprocessing(train_instances, test_instances):
    # Scaling the dataset
    scaler = MinMaxScaler()
    scaler.fit(train_instances)

    new_train_instances = scaler.transform(train_instances)
    new_test_instances  = scaler.transform(test_instances)

    pca = PCA(n_components=100)
    pca.fit(new_train_instances)

    new_train_instances = pca.transform(new_train_instances)
    new_test_instances  = pca.transform(new_test_instances)

    return new_train_instances, new_test_instances



def main():
    # Reading the dataset
    train_df, test_df = read_imdb_data(DATASET_PATH)
    #train_df = train_df.iloc[12000:12510, :]
    #test_df = test_df.iloc[11500:13500, :]

    print(train_df)
    print(train_df.columns)
    print(train_df['Rating'].value_counts())
    
    # Getting the entries with empty reviewText
    # because they are few we wont do anything
    print(train_df[train_df['Review'] == ''])
    
    # Getting dataset's instances and labels
    train_instances = train_df['Review']
    train_instances = data_preprocessing(train_instances)

    train_labels    = train_df['Rating'].apply(lambda x: int(x))
    train_labels    = convert_ratings_to_sentiments(train_labels)

    test_instances = test_df['Review']
    test_instances = data_preprocessing(test_instances)

    test_labels    = test_df['Rating'].apply(lambda x: int(x))
    test_labels    = convert_ratings_to_sentiments(test_labels)

    print(train_labels.value_counts())

    # Vectorization Part
    tfidf = TfidfVectorizer()

    tfidf_train_instances = tfidf.fit_transform(train_instances)
    tfidf_test_instances  = tfidf.transform(test_instances)

    cv = CountVectorizer(max_features=4000)

    cv_train_instances = cv.fit_transform(train_instances)
    cv_test_instances  = cv.transform(test_instances)

    instances           = list(train_instances) + list(test_instances)
    tokenized_instances = pd.Series(instances).apply(lambda x: x.split())

    # It will take some time
    model_w2v = train_and_get_word2vec_model(tokenized_instances)

    w2v_train_instances = extract_word2vec_instances(model_w2v, train_instances)
    w2v_test_instances  = extract_word2vec_instances(model_w2v, test_instances)

    del model_w2v
    print('Extracted instances')

    # Remove Constant features
    constant_filter = VarianceThreshold(threshold = 0.0002)

    constant_filter.fit(tfidf_train_instances)

    tfidf_train_instances = constant_filter.transform(tfidf_train_instances).toarray()
    tfidf_test_instances  = constant_filter.transform(tfidf_test_instances).toarray()

    constant_filter.fit(cv_train_instances)

    cv_train_instances = constant_filter.transform(cv_train_instances).toarray()
    cv_test_instances  = constant_filter.transform(cv_test_instances).toarray()
     
    print('Applied filtering')

    all_train_instances = [tfidf_train_instances, cv_train_instances, w2v_train_instances]
    all_test_instances  = [tfidf_test_instances, cv_test_instances, w2v_test_instances]
    instances_types     = ['Tf-Idf', 'CountVectorizer', 'Word2Vec']

    clfs       = [GaussianNB(), SVC(), SVC(C=10.0), SVC(C=50.0)]
    clfs_prep  = [None, svm_preprocessing, svm_preprocessing, svm_preprocessing]
    clfs_names = ['Naive Bayes', 'SVM with C=1.0', 'SVM with C=10.0', 'SVM with C=50.0']

    train_accuracies = []
    test_accuracies  = []
    train_f1_scores  = []
    test_f1_scores   = []

    for train_instances, test_instances in zip(all_train_instances, all_test_instances):
        curr_train_accs = []
        curr_test_accs  = []
        curr_train_f1s  = []
        curr_test_f1s   = []

        for clf, clf_prep in zip(clfs, clfs_prep):
            print(clf)
            if clf_prep != None:
                new_train_instances, new_test_instances = clf_prep(train_instances, test_instances)
            else:
                new_train_instances, new_test_instances = deepcopy(train_instances), deepcopy(test_instances)

            clf.fit(new_train_instances, train_labels)

            train_preds = clf.predict(new_train_instances)
            test_preds  = clf.predict(new_test_instances)

            curr_train_accs.append(accuracy_score(train_labels, train_preds))
            curr_test_accs.append(accuracy_score(test_labels, test_preds))

            curr_train_f1s.append(f1_score(train_labels, train_preds))
            curr_test_f1s.append(f1_score(test_labels, test_preds))

            del new_train_instances
            del new_test_instances
        
        train_accuracies.append(curr_train_accs)
        test_accuracies.append(curr_test_accs)
        train_f1_scores.append(curr_train_f1s)
        test_f1_scores.append(curr_test_f1s)

    print()
    print()
    print('Train Accuracies')
    print(pd.DataFrame(train_accuracies, index=instances_types, columns=clfs_names))
    print()
    print('Test Accuracies')
    print(pd.DataFrame(test_accuracies, index=instances_types, columns=clfs_names))
    print()
    print('Train F1-Scores')
    print(pd.DataFrame(train_f1_scores, index=instances_types, columns=clfs_names))
    print()
    print('Test F1-Scores')
    print(pd.DataFrame(test_f1_scores, index=instances_types, columns=clfs_names))

    return 0



if __name__ == "__main__":
    main()