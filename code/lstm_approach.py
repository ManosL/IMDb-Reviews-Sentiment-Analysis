import argparse

from collections import Counter
import sys

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

import gensim
import gensim.downloader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import PackedSequence, pack_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, WeightedRandomSampler

from utils import read_imdb_data, convert_ratings_to_sentiments
from utils import check_dir_exists, check_is_file, check_positive_float, check_positive_int

from sentiment_analysis_dataset import SentimentAnalysisDataset



device            = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATASET_PATH      = '../IMDB/aclImdb'



def collate_fn_padd(batch):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    instances, labels = zip(*batch)
    
    ## padd
    instances = [ torch.tensor(t).to(device) for t in instances ]
    instances = torch.nn.utils.rnn.pack_sequence(instances, enforce_sorted=False)
    
    if labels[0] != None:
        labels = torch.tensor(labels).to(device)

    return instances, labels



# This function contains the preprocessing that should be done in
# all our data regardless if they will be on train or test set
def data_preprocessing(instances, max_tokens, remove_stopwords, apply_stemming):
    processed_instances = pd.Series(instances)

    # Lowercase each review
    processed_instances = processed_instances.apply(lambda x: x.lower())

    # Replace sequent whitespaces with only one
    processed_instances = processed_instances.apply(lambda x: ' '.join(x.split()))

    # Tokenize each review and remove the stopwords
    processed_instances = processed_instances.apply(lambda x: word_tokenize(x))

    if remove_stopwords:
        stopwords_eng = set(stopwords.words('english'))
        processed_instances = processed_instances.apply(lambda x: [w for w in x if w not in stopwords_eng])

    # Stemming or Lemmatization
    if apply_stemming:
        stemmer = WordNetLemmatizer()
        processed_instances = processed_instances.apply(lambda x: [stemmer.lemmatize(w) for w in x])

    # Keep the first max_tokens tokens of each word
    
    if max_tokens != None:
        processed_instances = processed_instances.apply(lambda x: x[0:max_tokens] if len(x) > max_tokens else x)

    # Add special tokens that denote the start and end of review
    processed_instances = processed_instances.apply(lambda x: ['<SOR>'] + x + ['<EOR>'])

    # Each instance will contain a list of tokens
    return processed_instances



def extract_vocabulary_and_process_instances(train_instances, test_instances):
    vocabulary                = set(['<UNK>'])
    vocabulary_list           = ['<UNK>']
    processed_train_instances = []
    processed_test_instances  = []

    # Find the vocabulary from the train instances and convert them to list
    # of integers, i.e. Their index to the vocabulary
    for instance in list(train_instances):
        curr_sequence = []

        for token in instance:
            if token not in vocabulary:
                vocabulary.add(token)
                vocabulary_list.append(token)
            
            curr_sequence.append(vocabulary_list.index(token))

        processed_train_instances.append(curr_sequence)


    # According to vocabulary convert the test instances to list of
    # integers
    for instance in list(test_instances):
        curr_sequence = []

        for token in instance:
            if token not in vocabulary:
                curr_sequence.append(vocabulary_list.index('<UNK>'))
            else:
                curr_sequence.append(vocabulary_list.index(token))
        
        processed_test_instances.append(curr_sequence)

    return vocabulary_list, processed_train_instances, processed_test_instances



class LSTMSentimentModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_state_size):
        super().__init__()

        # Check if embeddings are getting trained
        self.embedding = nn.Embedding(vocab_size, embedding_size)

        self.lstm      = nn.LSTM(embedding_size, hidden_state_size, 2)

        self.output    = nn.Linear(hidden_state_size, 2)

        self.hidden_state_size = hidden_state_size
        return



    def forward(self, packed_tokens):
        embeddings = PackedSequence(self.embedding(packed_tokens.data), packed_tokens.batch_sizes,
                                        packed_tokens.sorted_indices, packed_tokens.unsorted_indices)

        out, _ = self.lstm(embeddings)

        out, out_lengths = pad_packed_sequence(out, batch_first=True)

        # Get the last item from each sequence
        out = out[torch.arange(out.shape[0]), out_lengths - 1]

        pred   = F.softmax(self.output(out), dim=1)
        
        return pred



class LSTMSentimentAnalysis:
    def __init__(self, epochs, batch_size, embedding_size=10, hidden_state_size=10, learning_rate=0.001):
        self.__epochs            = epochs
        self.__batch_size        = batch_size
        self.__embedding_size    = embedding_size
        self.__hidden_state_size = hidden_state_size
        self.__learning_rate     = learning_rate
        
        self.__vocabulary        = None
        self.__net               = None

        return



    def __preprocess_test_corpus(self, val_instances):
        processed_val_instances = []
        vocabulary_set = set(self.__vocabulary)

        for instance in list(val_instances):
            curr_sequence = []

            for token in instance:
                if token not in vocabulary_set:
                    curr_sequence.append('<UNK>')
                else:
                    curr_sequence.append(token)
            
            processed_val_instances.append(curr_sequence)

        return processed_val_instances



    def __get_dataloaders(self, train_instances, train_labels, val_instances, val_labels, sampler=None):
        train_dataset = SentimentAnalysisDataset(train_instances, train_labels, 
                                    self.__vocabulary, target_transform=torch.tensor)

        val_dataset   = SentimentAnalysisDataset(val_instances, val_labels, 
                                    self.__vocabulary, target_transform=torch.tensor)

        train_dataloader = DataLoader(train_dataset, batch_size=self.__batch_size, sampler=sampler, collate_fn=collate_fn_padd)
        val_dataloader   = DataLoader(val_dataset,   batch_size=self.__batch_size, collate_fn=collate_fn_padd)

        return train_dataloader, val_dataloader



    # Batch size will be 1 but consider to change it
    def fit(self, train_instances, train_labels, val_instances, val_labels):
        # Getting all characters
        self.__vocabulary = set(['<UNK>'])

        for instance in train_instances:
            for token in instance:
                self.__vocabulary.add(token)
        
        processed_val_instances = self.__preprocess_test_corpus(val_instances)
        
        self.__vocabulary = list(self.__vocabulary)

        # Getting sampler weights in order to fix class imbalance
        sampler_weights = []
        train_labels_counters = Counter(train_labels)

        for l in train_labels:
            assert(l in train_labels_counters.keys())
            sampler_weights.append(1 / train_labels_counters[l])
        
        sampler = WeightedRandomSampler(sampler_weights, len(sampler_weights))

        # Creating the DataLoaders for train and validation set
        train_dataloader, val_dataloader = self.__get_dataloaders(train_instances, train_labels,
                                                                processed_val_instances, val_labels, sampler)

        # Initialize the neural network
        self.__net = LSTMSentimentModel(len(self.__vocabulary), self.__embedding_size,
                                self.__hidden_state_size)
        
        self.__net.to(device)
        
        # Set loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.__net.parameters(), lr=self.__learning_rate)

        # we want to keep the best one, i.e the one with minimum validation loss
        min_validation_loss = None
        best_model          = None

        # Start training
        for epoch_num in range(self.__epochs):
            total_train_loss  = 0.0
            num_train_batches = 0

            for data in train_dataloader:
                inputs, labels = data

                # zero the gradient params of neural net
                optimizer.zero_grad()

                # get the output vector
                outputs = self.__net(inputs)
                loss = criterion(outputs, labels)

                # Calculating the gradients and updating the weights
                loss.backward()
                optimizer.step()

                total_train_loss  += loss.item()
                num_train_batches += 1

            total_train_loss = total_train_loss / num_train_batches

            # Calculation of normalized validation loss
            total_val_loss  = 0.0
            num_val_batches = 0
            
            self.__net.eval()

            for data in val_dataloader:
                inputs, labels = data

                # zero the gradient params of neural net
                optimizer.zero_grad()

                # get the output vector
                #state = state.detach()
                outputs = self.__net(inputs)

                loss = criterion(outputs, labels)

                total_val_loss  += loss.item()
                num_val_batches += 1
            
            total_val_loss = total_val_loss / num_val_batches

            print('Epoch', epoch_num, 'Train Loss(Normalized):', total_train_loss, 'Validation Loss(Normalized):', total_val_loss)
            sys.stdout.flush()

            self.__net.train()

            if (best_model == None) or (total_val_loss < min_validation_loss):
                best_model          = LSTMSentimentModel(len(self.__vocabulary), self.__embedding_size, self.__hidden_state_size)
                best_model.to(device)
                best_model.load_state_dict(self.__net.state_dict())

                min_validation_loss = total_val_loss


        self.__net = best_model
        
        return



    def predict(self, test_instances):
        self.__net.eval()
        processed_test_instances = self.__preprocess_test_corpus(test_instances)

        dataset = SentimentAnalysisDataset(processed_test_instances, None, 
                                    self.__vocabulary, target_transform=torch.tensor)

        dataloader = DataLoader(dataset, batch_size=self.__batch_size, collate_fn=collate_fn_padd)

        preds = []

        for data in dataloader:
            inputs, _ = data

            outputs = self.__net(inputs)
            outputs = torch.argmax(outputs, dim=1)

            preds += outputs.tolist()
                    
        self.__net.train()

        return preds


def main():
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser('description=LSTM Sentiment Clasifier')

    parser.add_argument('--dataset_path', type=check_dir_exists, default=DATASET_PATH,
                        help='path to the Product Reviews dataset')
    parser.add_argument('--epochs', type=check_positive_int, default=30,
                        help='number of epochs that our model will be trained')
    parser.add_argument('--batch_size', type=check_positive_int, default=256,
                        help='model\'s batch size')
    parser.add_argument('--learning_rate', type=check_positive_float, default=0.001,
                        help='Learning Rate of Adam optimizer')
    parser.add_argument('--embedding_size', type=check_positive_int, default=300,
                        help='Output size of embedding layer')
    parser.add_argument('--hidden_state_size', type=check_positive_int, default=128,
                        help='Output size of hidden vector of LSTM')
    parser.add_argument('--max_tokens', type=check_positive_int, default=500,
                        help='Max tokens I will take in each dataset instance')
    parser.add_argument('--remove_stopwords', action='store_true', default=False,
                        help='Whether to remove the stopwords from original corpus')
    parser.add_argument('--apply_stemming', action='store_true', default=False,
                        help='Whether to apply stemming to the corpus')
    parser.add_argument('--log', type=str, default=None,
                        help='Specify a log file to redirect stdout')

    args = parser.parse_args()
    
    if args.log != None:
        sys.stdout = open(args.log, 'w+')

    train_dataset, test_dataset = read_imdb_data(args.dataset_path)
    
    train_instances = data_preprocessing(train_dataset['Review'], args.max_tokens, 
                                    args.remove_stopwords, args.apply_stemming)

    train_labels    = train_dataset['Rating'].apply(lambda x: float(x))
    train_labels    = convert_ratings_to_sentiments(train_labels)

    test_instances  = data_preprocessing(test_dataset['Review'], args.max_tokens, 
                                    args.remove_stopwords, args.apply_stemming)

    test_labels     = test_dataset['Rating'].apply(lambda x: float(x))
    test_labels     = convert_ratings_to_sentiments(test_labels)

    # Remove empty elements
    train_instance_filter = train_instances.apply(lambda x: len(x) != 0)
    train_instances       = train_instances[train_instance_filter]
    train_labels          = train_labels[train_instance_filter]

    test_instance_filter  = test_instances.apply(lambda x: len(x) != 0)
    test_instances        = test_instances[test_instance_filter]
    test_labels           = test_labels[test_instance_filter]

    # Convert train and test instances and labels to lists
    train_instances = list(train_instances)
    train_labels    = list(train_labels)

    test_instances  = list(test_instances)
    test_labels     = list(test_labels)

    # Using some part of test set as validation set
    val_instances, test_instances, \
    val_labels, test_labels          = train_test_split(test_instances, test_labels, test_size=0.75, stratify=test_labels)

    print('Train instance num:', len(train_labels))
    sys.stdout.flush()

    wrapper = LSTMSentimentAnalysis(epochs=args.epochs, batch_size=args.batch_size, 
                                    embedding_size=args.embedding_size, 
                                    hidden_state_size=args.hidden_state_size,
                                    learning_rate=args.learning_rate)
    
    wrapper.fit(train_instances, train_labels, val_instances, val_labels)

    train_preds = list(wrapper.predict(train_instances))
    test_preds  = list(wrapper.predict(test_instances))
    
    """
    for r, p, l in zip(test_instances[0:100], test_preds[0:100], test_labels[0:100]):
        if p != l:
            print('Predicted wrong as', p, 'the', r)
    """
    
    print('Accuracy on Train Set:', accuracy_score(train_labels, train_preds))
    print('F1 on Train Set:', f1_score(train_labels, train_preds))

    print('Accuracy on Test Set:', accuracy_score(test_labels, test_preds))
    print('F1 on Test Set:', f1_score(test_labels, test_preds))
    print(confusion_matrix(test_labels, test_preds))

    print()
    test_instances = ['This is fucking disgusting.', 'After 1 month It does not work. I\'m dissapointed',
                        'Value for Money. I thought it would not work, but everything perfect.',
                        'I love it', 'Friends are the most dumb show I ever saw. Joey is lame.']
    
    test_instances = pd.Series(test_instances)
    test_instances = data_preprocessing(test_instances, args.max_tokens, args.remove_stopwords, args.apply_stemming)

    print(wrapper.predict(test_instances))

    sys.stdout.flush()

    if args.log != None:
        sys.stdout.close()
        sys.stdout = sys.__stdout__

    return 0



if __name__ == '__main__':
    main()
