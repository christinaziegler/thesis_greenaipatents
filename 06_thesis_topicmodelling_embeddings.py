# %%
# load json file as dictionary

import pandas as pd
import numpy as np
import re


from umap import UMAP
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
import bertopic
from hdbscan import HDBSCAN
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, PartOfSpeech
from bertopic.vectorizers import ClassTfidfTransformer


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

# precalculate embeddings: 
embedding_model_specter = SentenceTransformer("sentence-transformers/allenai-specter", revision=None)
embeddings_specter = embedding_model_specter.encode(abstracts, show_progress_bar=True)   

# Save embeddings
np.save('embeddings_specter_aifors.npy', embeddings_specter)
print("Saved embeddings specter aifors")

# %%
aifors_large = patents_text[patents_text['AIforS'] == 1]

def extract_first_300_words(text):
    words = re.findall(r'\b\w+\b', text)
    return ' '.join(words[:300])
abstracts = aifors_large['title'] + aifors_large['abstract'] + aifors_large["description"].fillna('').apply(extract_first_300_words)
abstracts = abstracts.apply(str).tolist()

# precalculate embeddings: 
embedding_model_specter = SentenceTransformer("sentence-transformers/allenai-specter", revision=None)
embeddings_specter = embedding_model_specter.encode(abstracts, show_progress_bar=True)   

# Save embeddings
np.save('embeddings_specter_aifors_large.npy', embeddings_specter)
print("Saved embeddings specter aifors_large")

# %%
sofai_large = patents_text[patents_text['SofAI'] == 1]

def extract_first_300_words(text):
    words = re.findall(r'\b\w+\b', text)
    return ' '.join(words[:300])
abstracts = sofai_large['title'] + sofai_large['abstract'] + sofai_large["description"].fillna('').apply(extract_first_300_words)
abstracts = abstracts.apply(str).tolist()

# precalculate embeddings: 
embedding_model_specter = SentenceTransformer("sentence-transformers/allenai-specter", revision=None)
embeddings_specter = embedding_model_specter.encode(abstracts, show_progress_bar=True)   

# Save embeddings
np.save('embeddings_specter_sofai_large.npy', embeddings_specter)
print("Saved embeddings specter sofai_large")

# %%
sofai = patents_text[(patents_text['SofAI'] == 1) & (patents_text['AIforS'] == 0)]

def extract_first_300_words(text):
    words = re.findall(r'\b\w+\b', text)
    return ' '.join(words[:300])
abstracts = sofai['title'] + sofai['abstract'] + sofai["description"].fillna('').apply(extract_first_300_words)
abstracts = abstracts.apply(str).tolist()

# precalculate embeddings: 
embedding_model_specter = SentenceTransformer("sentence-transformers/allenai-specter", revision=None)
embeddings_specter = embedding_model_specter.encode(abstracts, show_progress_bar=True)   

# Save embeddings
np.save('embeddings_specter_sofai.npy', embeddings_specter)
print("Saved embeddings specter sofai")
