
import json
import os
from joblib import load
import numpy as np
import pandas as pd
import pickle
import tiktoken
import pandas as pd

# WIRE THIS UP
#from config import Config
#app_config = Config()


# ================== #
#  Get Embeddings
# ================== #

#TODO: Change this to BERT instead of OPENAI
def get_embeddings(text=None, model=None):
    if model==None:
        model = load('./model/SentBERTmodel.pkl')

    return model.encode(text)


# ================== #
#  Load Embeddings
# ================== #
"""
def load_embeddings(fname: str) -> "dict[tuple[str, str], list[float]]":

    #Read the document embeddings and their keys from a CSV.

    #fname is the path to a CSV with exactly these named columns:
    #    "title", "heading", "0", "1", ... up to the length of the embedding vectors.
    
    df = pd.read_csv(fname, header=0)
    max_dim = max(
        [int(c) for c in df.columns if c != "title" and c != "heading"]
    )
    return {
        (r.title): [r[str(i)] for i in range(max_dim + 1)]
        for _, r in df.iterrows()
    }

"""






# ================== #
#  Tokenizer
# ================== #



# ================== #
#  Load Model
# ================== #