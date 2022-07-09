import torch
from torch.utils.data import Dataset

class SentimentAnalysisDataset(Dataset):
    def __init__(self, tokenized_instances, labels, vocabulary_list, 
                transform=None, target_transform=None):
        vocab_indexes = {}

        for i, w in zip(range(len(vocabulary_list)), vocabulary_list):
            vocab_indexes[w] = i

        self.transform = transform
        self.target_transform = target_transform

        # We will do something like extracting n-grams and convert them to tensors
        self.data   = []
        self.labels = labels

        for instance in tokenized_instances:
            curr_processed_instance = []

            for token in instance:
                curr_processed_instance.append(vocab_indexes[token])
            
            self.data.append(curr_processed_instance)
        
        return



    def __len__(self):
        return len(self.data)



    def __getitem__(self, idx):
        instance = self.data[idx]
        label    = self.labels[idx] if self.labels != None else None

        if self.transform:
            instance = self.transform(instance)

        if self.labels != None and self.target_transform:
            label = self.target_transform(label)

        return instance, label