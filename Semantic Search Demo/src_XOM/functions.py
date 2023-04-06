import numpy as np
import pandas as pd



# ================== #
#  Vector Similarity Calculation
# ================== #

def vector_similarity(x: "list[float]", y: "list[float]") -> float:
    """
    Returns the similarity between two vectors.

    Because embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
    """
    return np.dot(np.array(x), np.array(y))

# ================== #
#  Measure Similarity: Query vs Articles
# ================== #

def measure_embedding_similarity(
    query_embedding: str,
    article_embeddings
):
    """
    Compare the dot product of the query embedding against all of the pre-calculated document embeddings
    to measure the most relevant sections.
    """

    return [vector_similarity(query_embedding, article_embedding) for article_embedding in article_embeddings]

# ================== #
#  Slice & Order Articles by Similarity
# ================== #

def get_similar_texts(df, k):
    """
    Slice a dataframe on the top k results.  Sort the sliced dataframe descending on similarity score.

    If there are repeated results in top 5, keep them all.
    """
    response = df.nlargest(k, columns=['Similarity Score'],keep='all')
    response = response.sort_values(by='Similarity Score', ascending=False)
    return response