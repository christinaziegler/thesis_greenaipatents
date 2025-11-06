#!/usr/bin/env python
# coding: utf-8

# In[26]:


import yaml
import pandas as pd
import xml.etree.ElementTree as ET
import re
import numpy as np
import json
import time
import requests
from bs4 import BeautifulSoup
import html
import random
import seaborn as sns
import matplotlib.pyplot as plt
import textwrap
import os
# TQDM to Show Progress Bars #
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook

# SKLearn libraries for splitting sample and validation
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Additional Libraries that we are using only in this notebook
import torch
import gc

from collections import defaultdict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torch.optim import AdamW, Adamax, NAdam
from transformers import AutoTokenizer, AutoModel

from transformers import BertTokenizer, BertModel
from sklearn.metrics import f1_score
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold

import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import resample
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support
from imblearn.over_sampling import RandomOverSampler, SMOTE
from adapters import AutoAdapterModel
from transformers.adapters import AutoAdapterModel


# In[12]:


training_data = pd.read_csv("training_data_new.csv")


# In[13]:


binary_data = training_data[((training_data["SofAI"] == 1) & (training_data["AIforS"] == 0)) | ((training_data["SofAI"] == 0) & (training_data["AIforS"] == 1))]


# In[14]:


binary_data.loc[:, 'SustainableAI'] = np.where(binary_data['SofAI'] == 1, 1, 0)


# In[15]:


# Sample data (assuming training_data is your DataFrame and 'Narrow SofAI' is the target column)
# training_data = pd.read_csv('your_data.csv')  # Load your data

# Perform the stratified train-test split
train_df, test_df = train_test_split(
    binary_data, 
    test_size=0.2, 
    random_state=1,
    stratify=binary_data['SustainableAI'] # Stratify by the target column 
)

# Check the distribution in the splits
print(train_df['SustainableAI'].value_counts(normalize=True))
print(test_df['SustainableAI'].value_counts(normalize=True))


# # Classifier

# Try 1: Base Bert, no upsampling, no description

# In[16]:


# define function to report performance statistics
def classification_report(labels, predictions, filename="", save=False):
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    
    report = f"Accuracy: {accuracy:.2%}, "
    report += f"Precision: {precision:.2%}, "
    report += f"Recall: {recall:.2%}, "
    report += f"F1-score: {f1:.2%}"
    
    print(report)
    
    if save:
        with open(f'Classification_Report_{filename}.txt', 'w') as f:
            f.write(report)
    return report
    


# # General Code for parameter tuning/settings tuning

# In[17]:
performance_dict = {}
# Load the JSON file
with open('performance_dict_binary_20240620-122944.json', 'r') as f:
    performance_dict = json.load(f)




# # Part 2: Use classifier for Narrow AI for Sustainability

# In[35]:


def generate_filename(model_name, optimizer_name, labels, sampling, batchsize, dropoutrate1, dropoutrate2, suffix=''):
    return f"{labels}_{model_name}_{optimizer_name}_{sampling}_batch{batchsize}_dropout{dropoutrate1}and{dropoutrate2}_{suffix}"


class PatentDataset(Dataset):
    def __init__(self, df, abstract_column,
                 y_column, tokenizer=None): 
            self.df = df
            self.abstracts = df[abstract_column].apply(str).tolist()
            self.labels = df[y_column].tolist()  # Directly use the numerical labels
            self.tokenizer = tokenizer

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.abstracts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, padding='max_length', max_length=512, truncation=True, return_tensors="pt")
        return {'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten()}, label

class BertClassifier(nn.Module):
    def __init__(self, model_name = 'bert-base-uncased', freeze_layers=0, dropout_rate1=0.5, dropout_rate2=0.0):
        super(BertClassifier, self).__init__()
        if model_name == "allenai/specter2_aug2023refresh_base":
            self.bert = AutoAdapterModel.from_pretrained(model_name)
            adapter_name = self.bert.load_adapter("allenai/specter2_aug2023refresh_classification", source = "hf", set_active=True)
        else:
            self.bert = AutoModel.from_pretrained(model_name)
        self.dropout1 = nn.Dropout(dropout_rate1)
        self.dropout2 = nn.Dropout(dropout_rate2)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)
        # Improved method to freeze layers
        if freeze_layers > 0:
            for param in self.llm.parameters():
                param.requires_grad = False
                # Unfreeze specific layers
                for layer in [self.bert.embeddings, *self.bert.encoder.layer[:freeze_layers]]: #self.bert.encoder.layer[:freeze_layers]:
                    for param in layer.parameters():
                        param.requires_grad = True #so if 0 all remain frozen

            #layer_nums = [int(name.split('.')[2]) for name, param in self.bert.named_parameters() if name.startswith("encoder.layer") and name.split('.')[2].isdigit()]
            #max_freeze_layer = max(layer_nums) if layer_nums else 0
            # Freeze layers up to the specified number
            #for name, param in self.bert.named_parameters():
             #   if name.startswith("encoder.layer"):
             #       layer_num = int(name.split('.')[2])
             #       param.requires_grad = layer_num >= freeze_layers
             #   else:
             #       param.requires_grad = False if freeze_layers > 0 else True 
                    #Freezing everything if any encoder layer is frozen
                    
    def forward(self, input_ids, attention_mask):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_outputs.last_hidden_state  # Obtain the last hidden state
        pooled_output = bert_outputs.pooler_output  # Obtain the pooled output
        
        # Apply first dropout
        pooled_output = self.dropout1(pooled_output)
        
        # Optionally apply dropout to the sequence output before pooling or other operations
        # sequence_output = self.dropout1(sequence_output)
        
        # Apply the classifier
        logits = self.classifier(pooled_output)
        
        # Apply second dropout if needed before returning logits
        logits = self.dropout2(logits)
        #outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return logits

def oversample_data(dataset, indices):
    abstracts = np.array(dataset.abstracts)
    labels = np.array(dataset.labels)
    X_resampled, y_resampled = RandomOverSampler().fit_resample(abstracts[indices].reshape(-1, 1), labels[indices])
    X_resampled = X_resampled.flatten()
    return X_resampled, y_resampled


def oversample_data_smote(dataset, indices, tokenizer):
    abstracts = np.array(dataset.abstracts)
    labels = np.array(dataset.labels)
    
    # Tokenize and encode the abstracts
    encoded_inputs = dataset.tokenizer(abstracts[indices].tolist(), padding="max_length", max_length=512, truncation=True, return_tensors="pt")
    
    # Use the attention masks as features for SMOTE
    features = np.concatenate([encoded_inputs['input_ids'], encoded_inputs['attention_mask']], axis=1)
    
    smote = SMOTE()
    features_resampled, labels_resampled = smote.fit_resample(features, labels[indices])
    
    # Split the resampled features back to input_ids and attention_mask
    input_ids_resampled = features_resampled[:, :encoded_inputs['input_ids'].shape[1]]
    attention_mask_resampled = features_resampled[:, encoded_inputs['input_ids'].shape[1]:]
    
    return input_ids_resampled, attention_mask_resampled, labels_resampled



class ResampledPatentDataset(Dataset):
    def __init__(self, abstracts, labels, tokenizer=None):
        self.abstracts = abstracts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.abstracts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, padding='max_length', max_length=512, truncation=True, return_tensors="pt")
        return {'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten()}, label


class SMOTEPatentDataset(Dataset):
    def __init__(self, input_ids, attention_masks, labels):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {'input_ids': torch.tensor(self.input_ids[idx], dtype=torch.long),
                'attention_mask': torch.tensor(self.attention_masks[idx], dtype=torch.long)}, self.labels[idx]


def train_one_epoch(model, model_name, data_loader, optimizer, criterion, device):
    model.train()
    total_loss, total_accuracy = 0, 0

    progress_bar = tqdm(data_loader, desc='Training', leave=False)
    for data in progress_bar:
        inputs, labels = data
        inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to device
        labels = labels.to(device)
        outputs = model(**inputs)
        logits = outputs
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        total_accuracy += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(data_loader)
    avg_accuracy = total_accuracy / len(data_loader.dataset)
    return avg_loss, avg_accuracy

def cross_validate(model_name, df, abstract_column, y_column, sampling = "Oversample", dropoutrate1 = 0.5, dropoutrate2 = 0.0, tokenizer=None, device='cpu', epochs=10, n_splits=5, batchsize = 16, optimizer_cls=AdamW, save=False, multilabel="binary"):
    dataset = PatentDataset(df, abstract_column=abstract_column, y_column=y_column, tokenizer=tokenizer)
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold = 1
    results = []
    all_preds = []
    all_labels = []
    
    for train_index, val_index in kf.split(np.zeros(len(df[y_column].values)), df[y_column].values):
        print(f"Fold {fold}")
        if sampling == "Oversample":
            X_train_resampled, y_train_resampled = oversample_data(dataset, train_index)
            train_subset = ResampledPatentDataset(X_train_resampled, y_train_resampled, tokenizer=tokenizer)
        elif sampling == "SMOTE":
            input_ids_resampled, attention_mask_resampled, y_train_resampled = oversample_data_smote(dataset, train_index, tokenizer=tokenizer)
            train_subset = SMOTEPatentDataset(input_ids_resampled, attention_mask_resampled, y_train_resampled)    
        elif sampling == "None":
            train_subset = Subset(dataset, val_index)
        
        val_subset = Subset(dataset, val_index)
        
        train_data_loader = DataLoader(train_subset, batch_size=batchsize, shuffle=True)
        val_data_loader = DataLoader(val_subset, batch_size=batchsize, shuffle=False)

        if model_name == "SciBERT":
            model = BertClassifier('allenai/scibert_scivocab_uncased', dropout_rate1=dropoutrate1, dropout_rate2=dropoutrate2).to(device) 
        elif model_name == "BERT":
            model = BertClassifier(dropout_rate1=dropoutrate1, dropout_rate2=dropoutrate2).to(device)
        elif model_name == "PatentsBerta":
            model = BertClassifier('AI-Growth-Lab/PatentSBERTa', dropout_rate1=dropoutrate1, dropout_rate2=dropoutrate2).to(device)
        elif model_name == "Specter2":
            model = BertClassifier('allenai/specter2_aug2023refresh_base', dropout_rate1=dropoutrate1, dropout_rate2=dropoutrate2).to(device)
            #model.bert.load_adapter("allenai/specter2_aug2023refresh_classification", source = "hf", set_active=True)

        optimizer = optimizer_cls(model.parameters(), lr=5e-5)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            train_loss, train_accuracy = train_one_epoch(model, model_name, train_data_loader, optimizer, criterion, device)
            print(f"Fold {fold} | Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.4f}")
        
        # Evaluate the model on the validation fold
        model.eval()
        total_val_accuracy, total_val_loss = 0, 0
        fold_preds = []
        fold_labels = []
        
        with torch.no_grad():
            for data in val_data_loader:
                inputs, labels = data
                inputs = {k: v.to(device) for k, v in inputs.items()}
                labels = labels.to(device)
                outputs = model(**inputs)
                logits = outputs
                loss = criterion(logits, labels)
                total_val_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                total_val_accuracy += (predicted == labels).sum().item()
                
                fold_preds.extend(predicted.cpu().numpy())
                fold_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = total_val_loss / len(val_data_loader)
        avg_val_accuracy = total_val_accuracy / len(val_data_loader.dataset)
        fold_f1 = f1_score(fold_labels, fold_preds, average='binary')
        print(f"Fold {fold} | Validation Loss: {avg_val_loss:.4f} | Validation Accuracy: {avg_val_accuracy:.4f} | Validation F1: {fold_f1:.4f}")
        results.append((avg_val_loss, avg_val_accuracy, fold_f1))
        
        all_preds.extend(fold_preds)
        all_labels.extend(fold_labels)
        
        fold += 1

    avg_loss = sum([r[0] for r in results]) / len(results)
    avg_accuracy = sum([r[1] for r in results]) / len(results)
    avg_f1 = sum([r[2] for r in results]) / len(results)
    f1_range = max([r[2] for r in results]) - min([r[2] for r in results])
    print(f"Cross-validation results | Average Loss: {avg_loss:.4f} | Average Accuracy: {avg_accuracy:.4f} | Average F1: {avg_f1:.4f} | F1 Range: {f1_range:.4f}")

    # Calculate and display the confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Class 0', 'Class 1'])
    cm_display.plot(cmap='Blues')
    plt.show()
    if save:
        filename = generate_filename(model_name, optimizer_cls.__name__, multilabel, sampling, batchsize=batchsize, dropoutrate1=dropoutrate1, dropoutrate2=dropoutrate2, suffix="_confusion_matrix")
        plt.savefig(f"{filename}.png")

    return avg_loss, avg_accuracy, avg_f1, f1_range


class PatentTestDataset(Dataset):
    def __init__(self, df, abstract_column, tokenizer=None): 
            self.df = df
            self.abstracts = df[abstract_column].tolist()  # Directly use the numerical labels
            self.tokenizer = tokenizer

    def __len__(self):
        return len(self.abstracts)

    def __getitem__(self, idx):
        text = self.abstracts[idx]
        if not isinstance(text, str):
            text = str(text)
        encoding = self.tokenizer(text, padding='max_length', max_length=512, truncation=True, return_tensors="pt")
        return {'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten()}
    

def predict(model, model_name, data_loader, device):
    model = model.to(device)
    model.eval()
    predictions = []

    with torch.no_grad():
        for data in tqdm(data_loader, desc="Predicting"):
            inputs = {k: v.to(device) for k, v in data.items()}
            outputs = model(**inputs)
            logits = outputs
            _, predicted_labels = torch.max(logits, dim=1)
            predictions.extend(predicted_labels.cpu().numpy())
    return predictions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[36]:


df = train_df
optimizer_cls = AdamW
abstract_column = "text_descr1000"
multilabel = "binary"
batchsize = 16

for modelname in ["PatentsBerta", "Specter2", "BERT"]:
    for sampling in ["Oversample", "SMOTE"]:
        for dropoutrate1 in [0.5, 0.7]:
            for dropoutrate2 in [0.2]:
                base_filename = generate_filename(model_name=modelname, optimizer_name=optimizer_cls.__name__, labels=multilabel, 
                                                sampling=sampling, batchsize=batchsize, 
                                                dropoutrate1=dropoutrate1, dropoutrate2=dropoutrate2)
                print(f"Training {base_filename}")
                if modelname == "SciBERT":
                    tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
                elif modelname == "BERT":
                    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
                elif modelname == "PatentsBerta":
                    tokenizer = AutoTokenizer.from_pretrained('AI-Growth-Lab/PatentSBERTa')
                elif modelname == "Specter2":
                    tokenizer = AutoTokenizer.from_pretrained('allenai/specter2_aug2023refresh_base')

                train_dataset = PatentDataset(df, tokenizer=tokenizer, abstract_column=abstract_column, 
                                            y_column='SustainableAI')
                train_data_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
                avg_loss, avg_accuracy, avg_f1, f1_range = cross_validate(model_name=modelname, df=df, abstract_column=abstract_column, 
                                                                y_column="SustainableAI", sampling = sampling, 
                                                                dropoutrate1 = dropoutrate1, dropoutrate2=dropoutrate2, 
                                                                tokenizer=tokenizer, device=device, epochs=5, 
                                                                batchsize=batchsize, n_splits=5, optimizer_cls=optimizer_cls, 
                                                                multilabel=multilabel, save=True)
                performance_dict[base_filename] = (avg_loss, avg_accuracy, avg_f1, f1_range)

                # save performance_dict to a file with timestamp in the filename
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                with open(f'performance_dict_{multilabel}_{timestamp}.json', 'w') as f:
                    json.dump(performance_dict, f)

