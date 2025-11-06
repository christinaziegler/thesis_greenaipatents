#!/usr/bin/env python
# coding: utf-8

# In[26]:

import pandas as pd
import re
import numpy as np
import json
import time
import html
import random
import seaborn as sns
import matplotlib.pyplot as plt
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
from torch.optim import AdamW, Adamax, NAdam, SGD
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

from sklearn.datasets import make_classification
from sklearn.neighbors import NearestNeighbors

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import resample
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support
from imblearn.over_sampling import RandomOverSampler, SMOTE
#from adapters import AutoAdapterModel
#from transformers.adapters import AutoAdapterModel
from transformers.modeling_outputs import SequenceClassifierOutput

from transformers import LongformerForSequenceClassification
from torch.nn import BCEWithLogitsLoss
from transformers.models.longformer.modeling_longformer import LongformerPreTrainedModel, LongformerClassificationHead, LongformerConfig
from transformers import (
    LongformerPreTrainedModel,
    LongformerModel,
    LongformerConfig,
    AutoModelForSequenceClassification,
    #SequenceClassifierOutput,
    #BaseModelOutputWithPooling
)

        
from bert_multilabel_lossfunctions import ResampleLoss



def get_gpu_memory_usage():
    available_gpus = []
    for i in range(torch.cuda.device_count()):
        gpu_properties = torch.cuda.get_device_properties(i)
        reserved = torch.cuda.memory_reserved(i)
        allocated = torch.cuda.memory_allocated(i)
        free = gpu_properties.total_memory - (reserved - allocated)
        available_gpus.append((i, free))
    return available_gpus

def select_best_gpu():
    available_gpus = get_gpu_memory_usage()
    best_gpu = max(available_gpus, key=lambda x: x[1])
    return best_gpu[0]  # Return the index of the GPU with the most available memory

# Usage example:
best_gpu_index = select_best_gpu()
device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# In[12]:


training_data = pd.read_csv("training_data_new3.csv")

print(training_data.columns)

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

def multi_label_metrics(
    predictions, 
    labels, 
    ):
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    y_pred = np.zeros(probs.shape)
    y_true = labels
    y_pred[np.where(probs >= 0.5)] = 1
    f1_macro_average = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
    roc_auc = roc_auc_score(y_true, y_pred, average = 'macro')
    accuracy = accuracy_score(y_true, y_pred)
    # define dictionary of metrics to return
    metrics = {'f1': f1_macro_average,
               'roc_auc': roc_auc,
               'accuracy': accuracy}
    return metrics
    


# # General Code for parameter tuning/settings tuning

# In[17]:
performance_dict = {}


# # Part 2: Use classifier for Narrow AI for Sustainability

# In[35]:



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
        self.bert = AutoModelForSequenceClassification.from_pretrained(model_name, 
        problem_type = "multi_label_classification", num_labels = num_labels)
        self.dropout1 = nn.Dropout(dropout_rate1)
        self.dropout2 = nn.Dropout(dropout_rate2)
        # Improved method to freeze layers
        if freeze_layers > 0:
            for param in self.llm.parameters():
                param.requires_grad = False
                # Unfreeze specific layers
                for layer in [self.bert.embeddings, *self.bert.encoder.layer[:freeze_layers]]: #self.bert.encoder.layer[:freeze_layers]:
                    for param in layer.parameters():
                        param.requires_grad = True #so if 0 all remain frozen
    
          
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        logits = self.dropout1(logits)
        logits = self.dropout2(logits)

        return logits

def compute_combination_confusion_matrix(all_labels, all_predictions):
    # Calculate confusion matrix for all combinations of the two labels
    #combined_true = [f"{int(l[0])}{int(l[1])}" for l in all_labels]
    #combined_pred = [f"{int(p[0])}{int(p[1])}" for p in all_predictions]
    combined_labels = ["00", "01", "10", "11"]
    y_true_flat = [''.join(map(str, map(int, label))) for label in all_labels]
    y_pred_flat = [''.join(map(str, map(int, label))) for label in all_predictions]
    
    # Compute confusion matrix
    return confusion_matrix(y_true_flat, y_pred_flat, labels=['00', '01', '10', '11'])




def train_one_epoch(epoch, fold, model, model_name, loss_func, num_labels, data_loader, optimizer, device, accumulation_steps=4):
    model.train()
    total_loss, total_accuracy = 0, 0
    all_labels = []
    all_predictions = []
    progress_bar = tqdm(data_loader, desc='Training', leave=False)
    for data in progress_bar:
        inputs, labels = data
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)
        outputs = model(**inputs)
        logits = outputs
        loss = loss_func(logits.view(-1,num_labels),labels.type_as(logits).view(-1,num_labels))
        #oss_func(logits.view(-1,num_labels),logits.type_as(logits).view(-1,num_labels))
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
        torch.cuda.empty_cache()

    avg_loss = total_loss / len(data_loader)

    # Calculate F1-score
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    f1_0, f1_1 = f1_score(all_labels, all_predictions, average=None)
    
    cm = compute_combination_confusion_matrix(all_labels, all_predictions)
    
    # Create a confusion matrix display with custom labels
    labels = ['[0, 0]', '[0, 1]', '[1, 0]', '[1, 1]']
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    cm_display.plot(cmap='Blues')
    plt.show()
    plt.savefig(f"{multilabel}_imbalancedloss_{loss_func_name}_{model_name}_{abstract_column}_{optimizer_name}_batch{batchsize}_dropout{dropoutrate1}and{dropoutrate2}_confusionmatrix_fold{fold}_epoch{epoch}.png")
    return avg_loss, f1_0, f1_1
    


def cross_validate(model_name, df, loss_func_name, abstract_column, oversample = True, minority_class = "SofAI", dropoutrate1 = 0.5, dropoutrate2 = 0.0, tokenizer=None, device='cpu', epochs=10, n_splits=5, batchsize = 16, optimizer_cls=AdamW, multilabel="binary"):
    dataset = PatentDataset(df, abstract_column=abstract_column, tokenizer=tokenizer)
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    train_num = len(df)
    class_freq = np.array([df[col].sum() for col in label_columns], dtype=np.float32) 
    num_labels = len(label_columns)
    fold = 1
    results = []
    all_preds = []
    all_labels = []
    if loss_func_name == 'CBloss-ntr': # CB-NTR
        loss_func = ResampleLoss(reweight_func='CB', loss_weight=10.0,
                                focal=dict(focal=True, alpha=0.5, gamma=2),
                                logit_reg=dict(init_bias=0.05, neg_scale=2.0),
                                CB_loss=dict(CB_beta=0.9, CB_mode='by_class'),
                                class_freq=class_freq, train_num=train_num)
    
    elif loss_func_name == 'DBloss': # DB
        loss_func = ResampleLoss(reweight_func='rebalance', loss_weight=1.0,
                                focal=dict(focal=True, alpha=0.5, gamma=2),
                                logit_reg=dict(init_bias=0.05, neg_scale=2.0),
                                map_param=dict(alpha=0.1, beta=10.0, gamma=0.9), 
                                class_freq=class_freq, train_num=train_num)
    else:
        raise ValueError(f"Unsupported loss function: {loss_func_name}")
    
    for train_index, val_index in kf.split(np.zeros(len(df[minority_class].values)), df[minority_class].values):
        print(f"Fold {fold}")
        
        train_subset = Subset(dataset, train_index)
        
        val_subset = Subset(dataset, val_index)

        train_data_loader = DataLoader(train_subset, batch_size=batchsize, shuffle=True)
        val_data_loader = DataLoader(val_subset, batch_size=batchsize, shuffle=False)

        if model_name == "SciBERT":
            model = BertClassifier('allenai/scibert_scivocab_uncased', dropout_rate1=dropoutrate1, dropout_rate2=dropoutrate2).to(device) 
        elif model_name == "BERT":
            model = BertClassifier('bert-base-uncased', dropout_rate1=dropoutrate1, dropout_rate2=dropoutrate2).to(device)
        elif model_name == "PatentsBerta":
            model = BertClassifier('AI-Growth-Lab/PatentSBERTa', dropout_rate1=dropoutrate1, dropout_rate2=dropoutrate2).to(device)

        optimizer = optimizer_cls(model.parameters(), lr=5e-5)
        
 
        
        for epoch in range(epochs):
            train_loss, train_f1_0, train_f1_1 = train_one_epoch(epoch+1, fold, model, model_name, loss_func, num_labels, train_data_loader, optimizer, device, accumulation_steps = 4)
            print(f"Fold {fold} | Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train F1s: {train_f1_0:.4f}, {train_f1_1:.4f}")
        
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
                loss = loss_func(logits.view(-1,num_labels),labels.type_as(logits).view(-1,num_labels))
                total_val_loss += loss.item()
                # Convert logits to probabilities and then predictions
                probabilities = torch.sigmoid(logits)
                predicted_labels = (probabilities > 0.5).float()

                # Collect labels and predictions for evaluation
                fold_preds.extend(predicted_labels.cpu().detach().numpy())
                fold_labels.extend(labels.cpu().detach().numpy())


        avg_val_loss = total_val_loss / len(val_data_loader)
        fold_preds = np.array(fold_preds)
        fold_labels = np.array(fold_labels)
        fold_f1_macro = f1_score(fold_labels, fold_preds, average='macro')
        fold_f1_per_category = f1_score(fold_labels, fold_preds, average=None)
        fold_accuracy = accuracy_score(fold_labels, fold_preds)
        
        print(f"Fold {fold} | Validation Loss: {avg_val_loss:.4f} | Validation F1 (Macro): {fold_f1_macro:.4f} | Validation Accuracy: {fold_accuracy:.4f}")
        for idx, f1 in enumerate(fold_f1_per_category):
            print(f"Fold {fold} | Validation F1 for Category {idx}: {f1:.4f}")

        results.append((avg_val_loss, fold_f1_macro, fold_f1_per_category, fold_accuracy))

        all_preds.extend(fold_preds)
        all_labels.extend(fold_labels)
            
        fold += 1
    avg_loss = sum([r[0] for r in results]) / len(results)
    avg_f1_macro = sum([r[1] for r in results]) / len(results)
    avg_f1_per_category = np.mean([r[2] for r in results], axis=0)
    avg_accuracy = sum([r[3] for r in results]) / len(results)
    f1_range = max([r[1] for r in results]) - min([r[1] for r in results])
    
    print(f"Cross-validation results | Average Loss: {avg_loss:.4f} | Average F1 (macro): {avg_f1_macro:.4f} | Average Accuracy: {avg_accuracy:.4f} | F1 Range: {f1_range:.4f}")
    for idx, f1 in enumerate(avg_f1_per_category):
        print(f"Cross-validation results | Average F1 for Category {idx}: {f1:.4f}")

    cm = compute_combination_confusion_matrix(all_labels, all_preds)
    
    # Create a confusion matrix display with custom labels
    labels = ['[0, 0]', '[0, 1]', '[1, 0]', '[1, 1]']
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    cm_display.plot(cmap='Blues')
    plt.show()
    plt.savefig(f"{multilabel}_imbalancedloss_{loss_func_name}_{model_name}_{abstract_column}_{optimizer_name}_batch{batchsize}_dropout{dropoutrate1}and{dropoutrate2}_confusionmatrix_full.png")
    
    return avg_loss, avg_f1_macro, avg_f1_per_category, avg_accuracy, f1_range

       
    
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

def generate_filename(model_name, abstract_column, optimizer_name, loss_func_name, labels, batchsize, dropoutrate1, dropoutrate2, suffix=''):
    return f"{labels}_{model_name}_{abstract_column}_{optimizer_name}_{loss_func_name}_batch{batchsize}_dropout{dropoutrate1}and{dropoutrate2}_{suffix}"


# In[36]:

label_columns = ["SofAI", "AIforS"]  # Add your label columns here

df = train_df
abstract_column = "text_descr300"
multilabel = "multilabel"
optimizer_cls = AdamW
optimizer_name = "AdamW"

for modelname in ["BERT", "SciBERT", "PatentsBerta"]:
    for abstract_column in ["text_descr300"]:
        for dropoutrate1 in [0.2, 0.5]:
            for dropoutrate2 in [0.0, 0.2]:
                for batchsize in [16, 32]:
                    for loss_func_name in ["CBloss-ntr", "DBloss"]:
                        base_filename = generate_filename(model_name=modelname, abstract_column=abstract_column, optimizer_name=optimizer_cls.__name__, loss_func_name=loss_func_name, labels=multilabel, 
                                                        batchsize=batchsize, 
                                                        dropoutrate1=dropoutrate1, dropoutrate2=dropoutrate2)
                        print(f"Training {base_filename}")
                        if modelname == "SciBERT":
                            tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
                        elif modelname == "BERT":
                            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                        elif modelname == "PatentsBerta":
                            tokenizer = AutoTokenizer.from_pretrained('AI-Growth-Lab/PatentSBERTa')
                        train_dataset = PatentDataset(df, tokenizer=tokenizer, abstract_column=abstract_column)
                        train_data_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
                        avg_loss, avg_f1_macro, avg_f1_per_category, avg_accuracy, f1_range = cross_validate(model_name=modelname, df=df, abstract_column=abstract_column, 
                                                                        minority_class="SofAI", 
                                                                        dropoutrate1 = dropoutrate1, dropoutrate2=dropoutrate2, 
                                                                        tokenizer=tokenizer, device=device, epochs=5, 
                                                                        batchsize=batchsize, n_splits=5, optimizer_cls=optimizer_cls, 
                                                                        multilabel=multilabel, loss_func_name=loss_func_name)
                        performance_dict[base_filename] = {
                                                            "Average Loss": avg_loss,
                                                            "Average macro-F1": avg_f1_macro,
                                                            "F1 per Category 0": avg_f1_per_category[0],
                                                            "F1 per Category 1": avg_f1_per_category[1],
                                                            "Average Accuracy": avg_accuracy,
                                                            "F1 Range": f1_range
                                                        }
                        # save performance_dict to a file with timestamp in the filename
                        timestamp = time.strftime("%Y%m%d-%H%M%S")
                        with open(f'performance_dict_imbalancedloss_{multilabel}_baseline_rest_{timestamp}.json', 'w') as f:
                            json.dump(performance_dict, f)


