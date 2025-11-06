# %%
# load json file as dictionary
import json
import sys
import pandas as pd
import numpy as np
import re

from umap import UMAP
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
import bertopic
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, PartOfSpeech
from bertopic.vectorizers import ClassTfidfTransformer
import nbformat
import datamapplot

import seaborn as sns
from matplotlib import pyplot as plt
from adjustText import adjust_text
import matplotlib.patheffects as pe
import textwrap
import pickle 
from huggingface_hub import HfApi, HfFolder
from torch import cuda

from torch import bfloat16
import transformers

from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, TextGeneration

from bertopic import BERTopic


import itertools
import pandas as pd
import pickle
import torch 


torch.set_grad_enabled(False)

device = torch.device(f"cuda:{1}" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# topic model
topic_model = BERTopic.load("topicmodel_llama2_aifors_safetensors")
pd.set_option("display.max_columns", None)
#print(topic_model.get_topic_info())

# %%
patents_text = pd.read_csv("patents_text_topic_modelling2.csv")

# %%
patents_text['title'] = patents_text['title'].fillna('')
patents_text['abstract'] = patents_text['abstract'].fillna('')
patents_text['description'] = patents_text['description'].fillna('')
patents_text['claims'] = patents_text['claims'].fillna('')


aifors = patents_text[(patents_text['SofAI'] == 0) & (patents_text['AIforS'] == 1)]

def extract_first_300_words(text):
    words = re.findall(r'\b\w+\b', text)
    return ' '.join(words[:300])
abstracts = aifors['title'] + aifors['abstract'] + aifors["description"].fillna('').apply(extract_first_300_words)
abstracts = abstracts.apply(str).tolist()

# %%
# Load embeddings
embeddings_specter = np.load('embeddings_specter_aifors.npy')

# %%
def extract_cpc_prefix_set(cpc_string):
    cpc_string = cpc_string.strip("[]").replace("'", "")
    cpc_list = cpc_string.split(", ")
    cpc_prefixes = [code[:2] for code in cpc_list]
    return set(cpc_prefixes)

# Apply the function to each row in the 'cpc_codes_EPO' column
aifors['cpc_code_set'] = aifors['cpc_codes_EPO'].apply(extract_cpc_prefix_set)

# Convert sets to frozensets to make them hashable
frozenset_list = aifors['cpc_code_set'].apply(frozenset)

# Get the number of unique frozensets
unique_frozensets = frozenset_list.nunique()
# %%
# Create a mapping from each unique frozenset to a unique integer
unique_frozensets = frozenset_list.unique()
frozenset_to_int = {frozenset_: idx + 1 for idx, frozenset_ in enumerate(unique_frozensets)}

# Map the frozensets to integers and create a new column
aifors['unique_code'] = frozenset_list.map(frozenset_to_int)

# %%
categories = aifors["unique_code"]

topics, probs = topic_model.transform(abstracts, embeddings_specter)

new_topics = topic_model.reduce_outliers(abstracts, topics, strategy="embeddings", embeddings=embeddings_specter)
topic_model.update_topics(abstracts, topics=new_topics)


topics_to_merge = [[5, 7],
                    [15, 18, 21,22],
                    [14,19]]
topic_model.merge_topics(abstracts, topics_to_merge)




pd.reset_option("display.max_colwidth", None)
pd.set_option("display.max_columns", None)
topic_model.set_topic_labels({
    0: "Vehicle Gearbox and Power Transmission", 
    1: "Power Management of Electronic Devices", 
    2: 'Lighting',
    3: 'Power Conversion',
    4: "Wind Power Generation",
    5: "Process Control",
    6: "Internal Combustion Engine Control",
    7: "Waste Treatment",
    8: "Batteries and Charging",
    9: "Electric Vehicle Charging",
    10: "Manufacturing of Metallic Components",
    11: "Heat Pumps",
    12: "Wireless Communication",
    13: "Gas Turbines",
    14: "Fuel Cells",
    15: "Air Filtering",
    16: "Aircraft Systems",
    17: "Robotics"})
#topic_model.custom_labels_
topic_info = topic_model.get_topic_info()
first_three_columns = topic_info.iloc[:, [0,1,2,3]]
print(first_three_columns)

# Run the visualization with the original embeddings
fig = topic_model.visualize_document_datamap(abstracts, embeddings=embeddings_specter, custom_labels= True)
fig.write_html("datamap_aifors.html")
# with the reduced embeddings
#reduced_embeddings = UMAP(n_neighbors=15, n_components=2, min_dist=0.05, metric='cosine').fit_transform(embeddings_specter)
#fig = topic_model.visualize_document_datamap(abstracts, reduced_embeddings=reduced_embeddings, custom_labels=True)
#fig = topic_model.visualize_topics()
#fig.write_html("xy.html")

fig2 = topic_model.visualize_hierarchy(custom_labels=True)
fig2.write_html("hierarchy_visualisation_aifors.html")

list = ['Motor Drive Assembly for Vehicle', #Vehicle Gearbox and Power Transmission Systems
'Input Security Information Display Device Touch', #Power Management of Electronic Devices
'Power Conversion Systems', 
'Internal Combustion Engine Control Systems', #"Internal Combustion Engine Control Systems"
'Charging System for Electric Vehicle', 
'Lighting Device', 
'Process Control System', 
'Light Source', #merge with lighting device
'Waste Gas Treatment', # waste treatment
'Manufacturing of Metallic Components', 
'Wireless Communication Devices', 
'Thermal Pump System', 
'Fuel Cell System', 
'Gas Turbine Engine', 
'Battery Electrode Material', 
'Wind Turbine Control', #wind turbines
'Mobile Robot', #robotics
'Aircraft Flight Control System', #aircraft systems
'Wind Turbine Drivetrain System', #wind turbines
'Battery Charging System', 
'Air Filter Cartridge', # air filtering
'Wind Turbine Monitoring', #wind power generation
'Wind Power Generation System']
