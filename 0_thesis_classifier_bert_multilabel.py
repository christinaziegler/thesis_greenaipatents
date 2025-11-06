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
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

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
from transformers.modeling_outputs import SequenceClassifierOutput

# In[12]:


training_data = pd.read_csv("training_data_new.csv")
# Perform the stratified train-test split
train_df, test_df = train_test_split(
    training_data, 
    test_size=0.2, 
    random_state=1,
    stratify=training_data['SofAI'] # Stratify by the target column 
)

# In[13]:



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


# # Part 2: Use classifier for Narrow AI for Sustainability

# In[35]:


def generate_filename(model_name, optimizer_name, labels, batchsize, dropoutrate1, dropoutrate2, suffix=''):
    return f"{labels}_{model_name}_{optimizer_name}_batch{batchsize}_dropout{dropoutrate1}and{dropoutrate2}_{suffix}"


class PatentDataset(Dataset):
    def __init__(self, df, abstract_column,
                 label_columns = ["SofAI", "AIforS"], tokenizer=None): 
            self.df = df
            self.abstracts = df[abstract_column].apply(str).tolist()
            self.labels = df[label_columns].values.tolist()
            self.tokenizer = tokenizer

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.abstracts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, padding='max_length', max_length=512, truncation=True, return_tensors="pt")
        return {'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten()}, torch.tensor(label, dtype=torch.float32)

class BertClassifier(nn.Module):
    def __init__(self, model_name = 'bert-base-uncased', num_labels = 2, freeze_layers=0, dropout_rate1=0.5, dropout_rate2=0.0):
        super(BertClassifier, self).__init__()
        self.num_labels = num_labels
        if model_name == "allenai/specter2_aug2023refresh_base":
            self.bert = AutoAdapterModel.from_pretrained(model_name)
            adapter_name = self.bert.load_adapter("allenai/specter2_aug2023refresh_classification", source = "hf", set_active=True)
        else:
            self.bert = AutoModelForSequenceClassification.from_pretrained(model_name, 
            problem_type = "multi_label_classification", num_labels = num_labels)
        self.dropout1 = nn.Dropout(dropout_rate1)
        self.dropout2 = nn.Dropout(dropout_rate2)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
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
        if isinstance(bert_outputs, SequenceClassifierOutput):
            logits = bert_outputs.logits  # Directly get logits for sequence classification
        elif isinstance(bert_outputs, BaseModelOutputWithPooling):
            pooled_output = bert_outputs.pooler_output  # Get pooled output for models like BertModel
            pooled_output = self.dropout1(pooled_output)
            logits = self.classifier(pooled_output)
        else:
            raise NotImplementedError("Unsupported model output type")

        logits = self.dropout2(logits)
        return logits # Apply second dropout if needed before returning logits
        



def train_one_epoch(model, model_name, data_loader, optimizer, criterion, device):
    model.train()
    total_loss, total_accuracy = 0, 0
    all_labels = []
    all_predictions = []
    progress_bar = tqdm(data_loader, desc='Training', leave=False)
    for data in progress_bar:
        inputs, labels = data
        inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to device
        labels = labels.to(device)
        outputs = model(**inputs)
        logits = outputs
        loss = criterion(logits, labels.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
       # Convert logits to probabilities and then predictions
        probabilities = torch.sigmoid(logits)
        predicted_labels = (probabilities > 0.5).float()

        # Collect labels and predictions for F1-score calculation
        all_labels.extend(labels.cpu().detach().numpy())
        all_predictions.extend(predicted_labels.cpu().detach().numpy())

        progress_bar.set_postfix({'loss': total_loss / len(data_loader)})

    avg_loss = total_loss / len(data_loader)

    # Calculate micro F1-score
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    avg_f1 = f1_score(all_labels, all_predictions, average='micro')

    return avg_loss, avg_f1


def cross_validate(model_name, df, abstract_column, minority_class = "SofAI", dropoutrate1 = 0.5, dropoutrate2 = 0.0, tokenizer=None, device='cpu', epochs=10, n_splits=5, batchsize = 16, optimizer_cls=AdamW, save=False, multilabel="binary"):
    dataset = PatentDataset(df, abstract_column=abstract_column, tokenizer=tokenizer)
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold = 1
    results = []
    all_preds = []
    all_labels = []
    
    for train_index, val_index in kf.split(np.zeros(len(df[minority_class].values)), df[minority_class].values):
        print(f"Fold {fold}")
        train_subset = Subset(dataset, train_index)
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
        criterion = nn.BCEWithLogitsLoss()
        
        for epoch in range(epochs):
            train_loss, train_f1 = train_one_epoch(model, model_name, train_data_loader, optimizer, criterion, device)
            print(f"Fold {fold} | Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f}")
        
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
                loss = criterion(logits, labels.float())
                total_val_loss += loss.item()
                # Convert logits to probabilities and then predictions
                probabilities = torch.sigmoid(logits)
                predicted_labels = (probabilities > 0.5).float()

                # Collect labels and predictions for evaluation
                fold_preds.extend(predicted_labels.cpu().detach().numpy())
                fold_labels.extend(labels.cpu().detach().numpy())

        avg_val_loss = total_val_loss / len(val_data_loader)
        fold_f1 = f1_score(fold_labels, fold_preds, average='micro')
        fold_accuracy = accuracy_score(fold_labels, fold_preds)
        print(f"Fold {fold} | Validation Loss: {avg_val_loss:.4f} | Validation F1: {fold_f1:.4f} | Validation Accuracy: {fold_accuracy:.4f}")
        results.append((avg_val_loss, fold_f1, fold_accuracy))

        all_preds.extend(fold_preds)
        all_labels.extend(fold_labels)
            
        fold += 1

    avg_loss = sum([r[0] for r in results]) / len(results)
    avg_f1 = sum([r[1] for r in results]) / len(results)
    avg_accuracy = sum([r[2] for r in results]) / len(results)
    f1_range = max([r[1] for r in results]) - min([r[1] for r in results])
    print(f"Cross-validation results | Average Loss: {avg_loss:.4f} | Average F1: {avg_f1:.4f} | Average Accuracy: {avg_accuracy:.4f} | F1 Range: {f1_range:.4f}")

    # Calculate and display the confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm)
    cm_display.plot(cmap='Blues')
    plt.show()

    if save:
        filename = generate_filename(model_name, optimizer_cls.__name__, "multilabel", batchsize=batchsize, dropout_rate1=dropout_rate1, dropout_rate2=dropout_rate2, suffix="_confusion_matrix")
        plt.savefig(f"{filename}.png")

    return avg_loss, avg_f1, avg_accuracy, f1_range
        
    
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
multilabel = "multilabel"
batchsize = 16

for modelname in ["PatentsBerta"]:
    for abstract_column in ["text_descr1000"]:
        for dropoutrate1 in [0.5]:
            for dropoutrate2 in [0.2]:
                base_filename = generate_filename(model_name=modelname, optimizer_name=optimizer_cls.__name__, labels=multilabel, 
                                                batchsize=batchsize, 
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

                train_dataset = PatentDataset(df, tokenizer=tokenizer, abstract_column=abstract_column)
                train_data_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
                avg_loss, avg_accuracy, avg_f1, f1_range = cross_validate(model_name=modelname, df=df, abstract_column=abstract_column, 
                                                                minority_class="SofAI", 
                                                                dropoutrate1 = dropoutrate1, dropoutrate2=dropoutrate2, 
                                                                tokenizer=tokenizer, device=device, epochs=5, 
                                                                batchsize=batchsize, n_splits=5, optimizer_cls=optimizer_cls, 
                                                                multilabel=multilabel, save=True)
                performance_dict[base_filename] = (avg_loss, avg_accuracy, avg_f1, f1_range)

                # save performance_dict to a file with timestamp in the filename
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                with open(f'performance_dict_{multilabel}_{timestamp}.json', 'w') as f:
                    json.dump(performance_dict, f)

