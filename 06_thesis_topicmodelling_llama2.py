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



# %%
patents_text = pd.read_csv("patents_text_topic_modelling2.csv")

# %%
patents_text['title'] = patents_text['title'].fillna('')
patents_text['abstract'] = patents_text['abstract'].fillna('')
patents_text['description'] = patents_text['description'].fillna('')
patents_text['claims'] = patents_text['claims'].fillna('')

# %% [markdown]
# ## Full embeddings

# %%
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

print(aifors.cpc_code_set)
# Convert sets to frozensets to make them hashable
frozenset_list = aifors['cpc_code_set'].apply(frozenset)

# Get the number of unique frozensets
unique_frozensets = frozenset_list.nunique()

print("Number of unique sets of CPC code prefixes:", unique_frozensets)

# %%
# Create a mapping from each unique frozenset to a unique integer
unique_frozensets = frozenset_list.unique()
frozenset_to_int = {frozenset_: idx + 1 for idx, frozenset_ in enumerate(unique_frozensets)}

# Map the frozensets to integers and create a new column
aifors['unique_code'] = frozenset_list.map(frozenset_to_int)

# %%
categories = aifors["unique_code"]

# %%
# prevent stochastic behaviour
umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
reduced_embeddings = UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine', random_state=42).fit_transform(embeddings_specter)
# %%
# control number of topics
hdbscan_model = HDBSCAN(min_cluster_size=80, metric='euclidean', cluster_selection_method='eom', prediction_data=True)

# %%
# improve representation by preprocessing topic representations after documents are assigned
vectorizer_model = CountVectorizer(stop_words="english", min_df=2, ngram_range=(1, 2))

# %%


# Replace 'your_hugging_face_token' with your actual token
hf_token = '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'

# Save the token
HfFolder.save_token(hf_token)

# Alternatively, you can directly use the HfApi to login
api = HfApi(token=hf_token)


# %%


model_id = 'meta-llama/Llama-2-7b-chat-hf'
device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

print(device)




# set quantization configuration to load large model with less GPU memory
# this requires the `bitsandbytes` library

bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,  # 4-bit quantization
    bnb_4bit_quant_type='nf4',  # Normalized float 4
    bnb_4bit_use_double_quant=True,  # Second quantization after the first
    bnb_4bit_compute_dtype=bfloat16  # Computation type
)

# Llama 2 Tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

# Llama 2 Model
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    quantization_config=bnb_config,
    device_map='auto',
)
model.eval()


# Our text generator
generator = transformers.pipeline(
    model=model, tokenizer=tokenizer,
    task='text-generation',
    temperature=0.1,
    max_new_tokens=500,
    repetition_penalty=1.1
)

prompt = "Could you explain to me how 4-bit quantization works as if I am 5?"
res = generator(prompt)
print(res[0]["generated_text"])


# System prompt describes information given to all conversations
system_prompt = """
<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant for labeling topics.
<</SYS>>
"""


# Example prompt demonstrating the output we are looking for
example_prompt = """
I have a topic that contains the following documents:
- installation and process for purification of waste waterseparate volumes  of a tank are partitioned-off by walls  capable of movement. water level in each successive volume, remains constant when water is passed from one to another. to compensate for volume change, partitions move passively. an independent claim is included for the corresponding method of waste water treatment by moving the water in batches from one volume to the next for intervals of treatment. preferred features: each partition is solidly-attached to the tank by its base and sides. it is flexible, and dimensioned to bulge, compensating for changes in adjacent water volumes. partitions are watertight flexible sheets reaching up to the water level. they are over-sized to permit the required movement. floats keep the upper edges at the water level. the partition comprises reinforced plastic and moves to vary the surface area of the water volumes as required. partitions are constructed such that the bottom area of each volume remains constant. a recognized sequencing batch reactor (sbr) method is employed. the first volume functions as a pre-storage or feed tank . waste water is transferred by pumping to the second , where mixing, aeration, sedimentation and reaction phases take place. supernatant clarified water is transferred by pumping into the third volume which serves as an outlet tank. volume compensation between water inlet and outlet tanks, by change in area as indicated, are accompanied by a reduction in total times for sedimentation and water withdrawal.the invention relates to a wastewater treatment plant according to the preamble of patent claim and a corresponding wastewater treatment process according to the preamble of patent claim the invention is therefore based on wastewater treatment processes and plants in which wastewater is collected in a first separate container or water intake volume and from there is transferred in batches to a further water intake volume in which the wastewater batch is subjected to treatment cleaning before a clarified part is discharged from it such treatment processes can be referred to as sequential wastewater treatment processes which in particular include the so called aeration processes with damming operation as described in detail in the atv m leaflet atv regulations of the society for the promotion of wastewater technology e v this refers to processes for biological wastewater treatment in which activated sludge is used for biological wastewater treatment the biological cleaning processes and the separation of the activated sludge from the purified wastewater take place in one and the same treatment tank over a longer period of time in front of the actual clarification tank there is usually a pre storage tank in which the incoming wastewater is collected when a sufficient amount of wastewater has accumulated in the pre storage tank it is pumped into the clarification tank whereby the water level in the pre storage tank sinks while it rises accordingly in the clarification tank during this filling phase the wastewater to be treated is dammed up in the clarification tank and once the wastewater supply has been completed is subjected to a mixing phase during which anoxic and or anaerobic environmental conditions arise an aeration phase is followed by a settling phase during which activated sludge sediments in the lower area of the clarification tank finally when
- process and apparatus for purifying domestic waste watersgas (air) is introduced into a second tank  located inside a first tank . the wastewater passes through he second tank, at least partially. an independent claim is included for the corresponding plant treating domestic wastewater. preferred features: the gas is introduced low in the second tank, to rise up it. reduced pressure is caused in the second tank, inducing water into it, from the first. the water is recirculated between first and second tanks. a substrate immobilizing biomass on its surface, preferably moves freely about the second tank. before entering the first tank, the wastewater is mechanically pretreated in a first chamber. it is introduced from a second chamber into a third. a source of compressed gas  is connected by a line introducing the gas into the base of the second tank. a mushroom-shaped gas distributor  is located in the base of the second tank. the second tank has openings near its base, permitting water to flow in from the first. its top has a cover  with openings  passing water and gas into the first tank. both tanks are cylindrical. a base pedestal  permits adjustment of the second tank. first and second tanks forms part of a chamber. there is at least one further chamber for mechanical pretreatment. all may be arranged in the form of a small waste water treatment plant, essentially as described.the invention relates to a method for cleaning domestic waste water in which the waste water to be cleaned is introduced into a first container the invention further relates to a device for cleaning domestic waste water with a first container into which the waste water to be cleaned can be introduced the invention further relates to a device for cleaning domestic waste water which can be arranged in a container or basin of a small sewage treatment plant devices of this type are used to clean domestic waste water that has to be cleaned locally because the household in question is not connected to a sewage system the cleaned waste water can then be fed to a sprinkler or a receiving water body a method and device for cleaning domestic waste water of the type described at the beginning are known from din part small sewage treatment plants systems with waste water aeration application dimensioning design and testing june the waste water to be cleaned is introduced into an aeration tank provided with an aeration device the aeration device is used to introduce oxygen into the aeration tank and to circulate the resulting wastewater sludge mixture the aeration tank in which the wastewater is treated aerobically biologically is usually preceded by a mechanical cleaning stage in which settleable substances and floating substances are separated the invention is based on the object of providing a method and a device for cleaning domestic wastewater of the type described at the beginning with which a thorough biological cleaning of the wastewater is possible and which can also be used in existing small sewage treatment plants according to the invention this is achieved in a method for cleaning domestic wastewater of the type described at the beginning by introducing gas into a second
- biological installation and process for waste water purification in a biological installationa line  carrying recycled activated sludge back to the aeration stage, includes an ultrasonic reactor . an independent claim is included for the corresponding method of continuous treatment. preferred features: a similar variant plant is claimed, in which sludge recycle takes place from a septic tank outlet, with similar treatment in an ultrasonic reactor. at least % of the material from the activated sludge stage is treated over  hours, on a dry matter basis. the entire dry matter is subjected to ultrasound over at most, . days. fluid from the ultrasound treatment is sent the inlet of a grit chamber  or into an activated sludge tank. in the sludge stabilization stage, recirculated septic sludge is subjected to ultrasound. energy input is up to  watt hours/kg, based on dry material weight. frequency is - khz, at an intensity of .- w/cm<>. ultrasound treatment time lasts up to  seconds.the invention relates to a biological sewage treatment plant according to the preamble of claims and and to a method for cleaning waste water in a biological sewage treatment plant in which the activated sludge and or the digested sludge is treated with ultrasound in biological sewage treatment plants during sewage treatment so called primary sludge is produced by settling processes in primary clarifiers and biological sludge also known as secondary sludge is produced by metabolic processes of biodegradable substances caused by bacteria and other microorganisms the latter sludge is produced in aerobic anoxic and non aerobic process reactions such as aeration tanks and digestion tanks these biological processes take place slowly and have so far required a relatively large aeration tank and if the sewage treatment plant is equipped with one a large digestion tank although the microorganisms partially convert the sludge into biogas a large residual sludge volume remains which is disposed of as sewage sludge thereby causing considerable costs the purity levels that can be achieved in wastewater treatment are also unsatisfactory with the wastewater treatment processes known to date to remedy this it has already been proposed to treat part of the activated sludge with simultaneous aeration at intervals or to treat the secondary sludge with ultrasound in such a way that the microorganisms are destroyed by the high energy input for the ultrasound generation with the cell components of the killed microorganisms released serving as additional nutrients for the living microorganisms active in the loading tank although this allows the volume of excess sludge to be disposed of to be reduced the energy input is so high that this process is uneconomical the present invention is based on the object of specifying a biological sewage treatment plant and a process for cleaning wastewater in such

The topic is described by the following keywords: 'wastewater treatment, sewage treatment, water treatment, incineration, waste water, sewage, activated sludge, effluent, wastewater, dewatering'.

Based on the information about the topic above, please create a short label of this topic. Make sure you to only return the label and nothing more.

[/INST] Wastewater treatment
"""


# Our main prompt with documents ([DOCUMENTS]) and keywords ([KEYWORDS]) tags
main_prompt = """
[INST]
I have a topic that contains the following documents:
[DOCUMENTS]

The topic is described by the following keywords: '[KEYWORDS]'.

Based on the information about the topic above, please create a short label of this topic. Make sure you to only return the label and nothing more.
[/INST]
"""

prompt = system_prompt + example_prompt + main_prompt


# KeyBERT
keybert = KeyBERTInspired()

# MMR
mmr = MaximalMarginalRelevance(diversity=0.3)

# Text generation with Llama 2
llama2 = TextGeneration(generator, prompt=prompt)

# All representation models
representation_model = {
    "KeyBERT": keybert,
    "Llama2": llama2,
    "MMR": mmr,
}


embedding_model_specter = SentenceTransformer("sentence-transformers/allenai-specter", revision=None)

topic_model = BERTopic(
  # Sub-models
  embedding_model=embedding_model_specter,
  umap_model=umap_model,
  hdbscan_model=hdbscan_model,
  representation_model=representation_model,

  # Hyperparameters
  top_n_words=10,
  verbose=True
)

# Train model
topics, probs = topic_model.fit_transform(abstracts, embeddings_specter, y=categories)

# Show topics
topic_model.get_topic_info()

llama2_labels = [label[0][0].split("\n")[0] for label in topic_model.get_topics(full=True)["Llama2"].values()]
topic_model.set_topic_labels(llama2_labels)

with open('rep_docs.pickle', 'wb') as handle:
    pickle.dump(topic_model.representative_docs_, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('reduced_embeddings.pickle', 'wb') as handle:
    pickle.dump(reduced_embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Save the BERTopic model using pickle
topic_model.save("topicmodel_llama2_aifors_safetensors", serialization="safetensors", save_ctfidf=True, save_embedding_model=True)

# Define colors for the visualization to iterate over
colors = itertools.cycle(['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000'])
color_key = {str(topic): next(colors) for topic in set(topic_model.topics_) if topic != -1}

# Prepare dataframe and ignore outliers
df = pd.DataFrame({"x": reduced_embeddings[:, 0], "y": reduced_embeddings[:, 1], "Topic": [str(t) for t in topic_model.topics_]})
df["Length"] = [len(doc) for doc in abstracts]
df = df.loc[df.Topic != "-1"]
df = df.loc[(df.y > -10) & (df.y < 10) & (df.x < 10) & (df.x > -10), :]
df["Topic"] = df["Topic"].astype("category")

# Get centroids of clusters
mean_df = df.groupby("Topic").mean().reset_index()
mean_df.Topic = mean_df.Topic.astype(int)
mean_df = mean_df.sort_values("Topic")



fig = plt.figure(figsize=(10, 10))
sns.scatterplot(data=df, x='x', y='y', hue="Topic", palette=color_key, alpha=0.4, sizes=(0.4, 10), size="Length")

# Annotate top 50 topics
texts, xs, ys = [], [], []
for row in mean_df.iterrows():
  topic = row[1]["Topic"]
  name = textwrap.fill(topic_model.custom_labels_[int(topic)], 20)

  if int(topic) <= 50:
    xs.append(row[1]["x"])
    ys.append(row[1]["y"])
    texts.append(plt.text(row[1]["x"], row[1]["y"], name, size=10, ha="center", color=color_key[str(int(topic))],
                          path_effects=[pe.withStroke(linewidth=0.5, foreground="black")]
                          ))

# Adjust annotations such that they do not overlap
adjust_text(texts, x=xs, y=ys, time_lim=1, force_text=(0.01, 0.02), force_static=(0.01, 0.02), force_pull=(0.5, 0.5))
plt.axis('off')
plt.legend('', frameon=False)
plt.show()

fig.savefig('topic_model_aifors_scatterplot.png', dpi=300, bbox_inches='tight')
