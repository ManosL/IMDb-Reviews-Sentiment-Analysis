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

from torchinfo import summary

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import PackedSequence, pack_sequence, pad_packed_sequence
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

import transformers
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import get_linear_schedule_with_warmup

from utils import read_imdb_data, convert_ratings_to_sentiments
from utils import check_dir_exists, check_is_file, check_positive_float, check_positive_int



device            = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATASET_PATH      = '../IMDB/aclImdb'


# Supress huggingFace warnings
transformers.logging.set_verbosity_error()



class BERTSentimentAnalysis:
    def __init__(self, epochs, batch_size, pretrained_model, max_tokens=256):
        self.__epochs            = epochs
        self.__batch_size        = batch_size
        self.__max_tokens        = max_tokens
        self.__pretrained_model  = pretrained_model

        self.__tokenizer         = BertTokenizer.from_pretrained(pretrained_model, 
                                                                    do_lower_case=True
                                                                )
                                                            
        self.__net               = None

        return



    def __create_dataloader(self, instances, labels, sampler=None):
        encoded_instances = self.__tokenizer.batch_encode_plus(instances, 
                                                                add_special_tokens=True,
                                                                return_attention_mask=True,
                                                                pad_to_max_length=True,
                                                                max_length=self.__max_tokens,
                                                                truncation=True,
                                                                padding='max_length',
                                                                return_tensors='pt'
                                                            )
        
        input_ids       = encoded_instances['input_ids']
        attention_masks = encoded_instances['attention_mask']

        if labels == None:
            dataset = TensorDataset(input_ids, attention_masks)
        else:
            labels = torch.tensor(labels)
            dataset = TensorDataset(input_ids, attention_masks, labels)

        dataloader = DataLoader(dataset, batch_size=self.__batch_size, sampler=sampler)

        return dataloader



    def __get_dataloaders(self, train_instances, train_labels, val_instances, val_labels, sampler=None):
        train_dataloader = self.__create_dataloader(train_instances, train_labels, sampler)
        val_dataloader   = self.__create_dataloader(val_instances, val_labels)

        return train_dataloader, val_dataloader



    # Batch size will be 1 but consider to change it
    def fit(self, train_instances, train_labels, val_instances, val_labels):
        # Getting the number of labels and initialize the BERT model
        num_labels = len(set(list(train_labels)))

        self.__net = BertForSequenceClassification.from_pretrained(self.__pretrained_model, 
                                                                    num_labels=num_labels,
                                                                    output_attentions=False, 
                                                                    output_hidden_states=False
                                                                )

        self.__net.to(device)
        
        # Getting sampler weights in order to fix class imbalance
        sampler_weights = []
        train_labels_counters = Counter(train_labels)

        for l in train_labels:
            assert(l in train_labels_counters.keys())
            sampler_weights.append(1 / train_labels_counters[l])
        
        sampler = WeightedRandomSampler(sampler_weights, len(sampler_weights))

        # Creating the DataLoaders for train and validation set
        train_dataloader, val_dataloader = self.__get_dataloaders(train_instances, train_labels,
                                                                val_instances, val_labels, sampler)
        
        # Set loss function, optimizer and scheduler
        optimizer = optim.AdamW(self.__net.parameters(), lr=1e-5, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                    num_warmup_steps=0, 
                                                    num_training_steps=len(train_dataloader) * self.__epochs
                                                )

        # we want to keep the best one, i.e the one with minimum validation accuracy
        min_validation_acc  = None
        best_model          = None

        # Start training
        for epoch_num in range(self.__epochs):
            total_train_loss  = 0.0
            num_train_batches = 0

            for data in train_dataloader:
                input_ids, attention_mask, labels = data[0].to(device), data[1].to(device), data[2].to(device)

                # zero the gradient params of neural net
                optimizer.zero_grad()

                # get the output vector
                outputs = self.__net(input_ids=input_ids, attention_mask=attention_mask,
                                    labels=labels)

                loss    = outputs[0]

                # Calculating the gradients and updating the weights
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.__net.parameters(), 1.0)

                optimizer.step()
                scheduler.step()

                total_train_loss  += loss.item()
                num_train_batches += 1

            total_train_loss = total_train_loss / num_train_batches

            # Calculation of normalized validation loss
            total_val_loss  = 0.0
            num_val_batches = 0
            
            self.__net.eval()
            val_preds  = []
            val_labels = []

            for data in val_dataloader:
                input_ids, attention_mask, labels = data[0].to(device), data[1].to(device), data[2].to(device)

                # zero the gradient params of neural net
                optimizer.zero_grad()

                # get the output vector
                #state = state.detach()
                outputs = self.__net(input_ids=input_ids, attention_mask=attention_mask,
                                    labels=labels)

                loss             = outputs[0]
                total_val_loss  += loss.item()
                num_val_batches += 1

                logits      = outputs.logits
                val_preds  += logits.argmax(dim=1).tolist()
                val_labels += labels.tolist()
            
            total_val_loss = total_val_loss / num_val_batches

            print('Epoch', epoch_num, 'Train Loss(Normalized):', total_train_loss, 'Validation Loss(Normalized):', total_val_loss)
            sys.stdout.flush()

            self.__net.train()
            
            val_accuracy = accuracy_score(val_labels, val_preds)

            if (best_model == None) or (val_accuracy < min_validation_acc):
                best_model          = BertForSequenceClassification.from_pretrained(self.__pretrained_model, 
                                                                    num_labels=num_labels,
                                                                    output_attentions=False, 
                                                                    output_hidden_states=False
                                                                )                

                best_model.to(device)
                best_model.load_state_dict(self.__net.state_dict())

                min_validation_acc = val_accuracy


        self.__net = best_model
        
        return



    def predict(self, test_instances):
        self.__net.eval()
        dataloader = self.__create_dataloader(test_instances, labels=None)

        preds = []

        for data in dataloader:
            input_ids, attention_mask = data[0].to(device), data[1].to(device)

            with torch.no_grad():
                outputs = self.__net(input_ids=input_ids, attention_mask=attention_mask)

            logits  = outputs.logits
            preds  += logits.argmax(dim=1).tolist()

        self.__net.train()

        return preds


def main():
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser('description=BERT Sentiment Clasifier')

    parser.add_argument('--dataset_path', type=check_dir_exists, default=DATASET_PATH,
                        help='path to the Product Reviews dataset')
    parser.add_argument('--pretrained_model', type=str, default='prajjwal1/bert-small',
                        help='name of pretrained model in HuggingFace')
    parser.add_argument('--epochs', type=check_positive_int, default=30,
                        help='number of epochs that our model will be trained')
    parser.add_argument('--batch_size', type=check_positive_int, default=32,
                        help='model\'s batch size')
    parser.add_argument('--max_tokens', type=check_positive_int, default=256,
                        help='Max tokens BERT model will take')
    parser.add_argument('--log', type=str, default=None,
                        help='Specify a log file to redirect stdout')

    args = parser.parse_args()
    
    if args.log != None:
        sys.stdout = open(args.log, 'w+')

    train_dataset, test_dataset = read_imdb_data(args.dataset_path)
    
    train_instances = train_dataset['Review']

    train_labels    = train_dataset['Rating'].apply(lambda x: float(x))
    train_labels    = convert_ratings_to_sentiments(train_labels)

    test_instances  = test_dataset['Review']

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

    wrapper = BERTSentimentAnalysis(epochs=args.epochs, batch_size=args.batch_size,
                                    pretrained_model=args.pretrained_model, max_tokens=args.max_tokens)
    
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
    # Trying on some random instances
    test_instances = ['This is fucking disgusting.', 'After 1 month It does not work. I\'m dissapointed',
                        'Value for Money. I thought it would not work, but everything perfect.',
                        'I love it', 'Friends are the most dumb show I ever saw. Joey is lame.']
    
    test_instances = pd.Series(test_instances)

    print(wrapper.predict(test_instances))

    sys.stdout.flush()

    if args.log != None:
        sys.stdout.close()
        sys.stdout = sys.__stdout__

    return 0



if __name__ == '__main__':
    main()
