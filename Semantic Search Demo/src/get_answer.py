import numpy as np
import pandas as pd
from joblib import load

#from src.embed import get_embedding


def get_embeddings(text=None, model=None):
    if model==None:
        model = load('./model/SentBERTmodel.pkl')

    return model.encode(text)


# ================== #
#  Calculate Vector Similarity
# ================== #

def vector_similarity(x: "list[float]", y: "list[float]") -> float:
    """
    Returns the similarity between two vectors.

    Because Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
    """
    # TODO check this docstring ^
    return np.dot(np.array(x), np.array(y))


# ================== #
#  Order Chunks by Similarity
# ================== #

def order_chunks_by_similarity(
    query: str, 
    embeddings,
) -> list[(float, (str, str))]:
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections.

    Return the list of document sections, sorted by relevance in descending order.
    """
    query_embedding = get_embeddings(query)

    document_similarities = sorted(
        [
            (vector_similarity(query_embedding, doc_embedding), doc_index)
            for doc_index, doc_embedding in embeddings
        ],
        reverse=True,
    )

    return document_similarities


# ================== #
# 
# ================== #



# ================== #
# 
# ================== #




# ================== #
# 
# ================== #