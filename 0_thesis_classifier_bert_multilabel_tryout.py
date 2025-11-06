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

from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint


def print_gpu_memory():
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 3} GB")
    print(f"Cached: {torch.cuda.memory_reserved() / 1024 ** 3} GB")

print_gpu_memory()
torch.cuda.empty_cache()
print_gpu_memory()
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

def multi_label_metrics(
    predictions, 
    labels, 
    ):
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    y_pred = np.zeros(probs.shape)
    y_true = labels
    y_pred[np.where(probs >= 0.5)] = 1
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
    accuracy = accuracy_score(y_true, y_pred)
    # define dictionary of metrics to return
    metrics = {'f1': f1_micro_average,
               'roc_auc': roc_auc,
               'accuracy': accuracy}
    return metrics
    


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
        encoding = self.tokenizer(text, padding='max_length', max_length=4096, truncation=True, return_tensors="pt")
        return {'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten()}, torch.tensor(label, dtype=torch.float32)


class LongformerForMultiLabelSequenceClassification(LongformerPreTrainedModel):
    """
    We instantiate a class of LongFormer adapted for a multilabel classification task. 
    This instance takes the pooled output of the LongFormer based model and passes it through a classification head. We replace the traditional Cross Entropy loss with a BCE loss that generate probabilities for all the labels that we feed into the model.
    """

    def __init__(self, config):
        super(LongformerForMultiLabelSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.longformer = LongformerModel(config)
        self.classifier = LongformerClassificationHead(config)
        self.init_weights()
        
    def forward(self, input_ids=None, attention_mask=None, global_attention_mask=None, 
                token_type_ids=None, position_ids=None, inputs_embeds=None, 
                labels=None):
        
        # create global attention on sequence, and a global attention token on the `s` token
        # the equivalent of the CLS token on BERT models. This is taken care of by HuggingFace
        # on the LongformerForSequenceClassification class
        if global_attention_mask is None:
            global_attention_mask = torch.zeros_like(input_ids)
            global_attention_mask[:, 0] = 1
        
        # pass arguments to longformer model
        outputs = self.longformer(
            input_ids = input_ids,
            attention_mask = attention_mask,
            global_attention_mask = global_attention_mask,
            token_type_ids = token_type_ids,
            position_ids = position_ids)
        
        # if specified the model can return a dict where each key corresponds to the output of a
        # LongformerPooler output class. In this case we take the last hidden state of the sequence
        # which will have the shape (batch_size, sequence_length, hidden_size). 
        sequence_output = outputs['last_hidden_state']
        
        # pass the hidden states through the classifier to obtain thee logits
        logits = self.classifier(sequence_output)
        outputs = (logits,) + outputs[2:]
        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            labels = labels.float()
            loss = loss_fct(logits.view(-1, self.num_labels), 
                            labels.view(-1, self.num_labels))
            #outputs = (loss,) + outputs
            outputs = (loss,) + outputs
        
        
        return outputs

class LongformerClassifier(LightningModule):
    def __init__(self, model_name='allenai/longformer-base-4096', num_labels=2):
        super().__init__()
        self.model = LongformerForMultiLabelSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs['input_ids'], attention_mask=inputs['attention_mask'])
        loss = nn.BCEWithLogitsLoss()(outputs.logits, labels.float())
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=5e-5)
        return optimizer



def train_one_epoch(model, model_name, data_loader, optimizer, criterion, device, accumulation_steps=4):
    model.train()
    total_loss, total_accuracy = 0, 0
    all_labels = []
    all_predictions = []
    progress_bar = tqdm(data_loader, desc='Training', leave=False)
    for data in progress_bar:
        inputs, labels = data
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)
        with torch.cuda.amp.autocast():
            outputs = model(**inputs)
            logits = outputs
            loss = criterion(logits, labels.float())
        checkpointed_backward = torch.utils.checkpoint.checkpoint_backward(loss, inputs)
        scaler.scale(checkpointed_backward).backward()
        

        if (step + 1) % accumulation_steps == 0 or (step + 1) == len(data_loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            torch.cuda.empty_cache()  # Clear GPU cache after each accumulation step

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

        
        config = LongformerConfig.from_pretrained('allenai/longformer-base-4096', num_labels=2)
        model = LongformerForMultiLabelSequenceClassification(config).to(device)
            
        optimizer = optimizer_cls(model.parameters(), lr=5e-5)
        criterion = nn.BCEWithLogitsLoss()
        scaler = torch.cuda.amp.GradScaler()
        
        for epoch in range(epochs):
            train_loss, train_f1 = train_one_epoch(model, model_name, train_data_loader, optimizer, criterion, device, accumulation_steps = 4)
            print(f"Fold {fold} | Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f}")
        
        # Evaluate the model on the validation fold
        model.eval()
        total_val_accuracy, total_val_loss = 0, 0
        fold_preds = []
        fold_labels = []
        
        with torch.no_grad():
            for data in val_data_loader:
                inputs, labels = data
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
        fold_preds = np.array(fold_preds)
        fold_labels = np.array(fold_labels)
        fold_f1_micro = f1_score(fold_labels, fold_preds, average='micro')
        fold_f1_per_category = f1_score(fold_labels, fold_preds, average=None)
        fold_accuracy = accuracy_score(fold_labels, fold_preds)
        
        print(f"Fold {fold} | Validation Loss: {avg_val_loss:.4f} | Validation F1 (Micro): {fold_f1_micro:.4f} | Validation Accuracy: {fold_accuracy:.4f}")
        for idx, f1 in enumerate(fold_f1_per_category):
            print(f"Fold {fold} | Validation F1 for Category {idx}: {f1:.4f}")

        results.append((avg_val_loss, fold_f1_micro, fold_f1_per_category, fold_accuracy))

        all_preds.extend(fold_preds)
        all_labels.extend(fold_labels)
            
        fold += 1

    avg_loss = sum([r[0] for r in results]) / len(results)
    avg_f1_micro = sum([r[1] for r in results]) / len(results)
    avg_f1_per_category = np.mean([r[2] for r in results], axis=0)
    avg_accuracy = sum([r[3] for r in results]) / len(results)
    f1_range = max([r[1] for r in results]) - min([r[1] for r in results])
    
    print(f"Cross-validation results | Average Loss: {avg_loss:.4f} | Average F1 (Micro): {avg_f1_micro:.4f} | Average Accuracy: {avg_accuracy:.4f} | F1 Range: {f1_range:.4f}")
    for idx, f1 in enumerate(avg_f1_per_category):
        print(f"Cross-validation results | Average F1 for Category {idx}: {f1:.4f}")

    cm = confusion_matrix(all_labels, all_preds)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm)
    cm_display.plot(cmap='Blues')
    plt.show()

    if save:
        filename = generate_filename(model_name, optimizer_cls.__name__, "multilabel", batchsize=batchsize, dropout_rate1=dropoutrate1, dropout_rate2=dropoutrate2, suffix="_confusion_matrix")
        plt.savefig(f"{filename}.png")

    return avg_loss, avg_f1_micro, avg_f1_per_category, avg_accuracy, f1_range

       


# In[36]:


def main():
    seed_everything(42)  # Ensure reproducibility

    # Define your dataset and dataloaders
    tokenizer = AutoTokenizer.from_pretrained('allenai/longformer-base-4096')
    abstract_column = "text_descr1000"
    df = train_df


    dataset = PatentDataset(df, tokenizer=tokenizer, abstract_column=abstract_column)
    
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Initialize your LightningModule
    model = LongformerClassifier()

    # Configure PyTorch Lightning Trainer
    trainer = Trainer(precision=16, amp_level='O2')  # Enable mixed precision training

    # Train the model
    trainer.fit(model, train_loader)

if __name__ == '__main__':
    main()

