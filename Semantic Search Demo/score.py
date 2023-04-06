# Import Packages
import os
import json
import numpy as np
import pandas as pd
import logging
from joblib import load

# Functions from src
from src_XOM import vector_similarity, measure_embedding_similarity, get_similar_texts


LOGGER = logging.getLogger(__name__)


def init():
    global BERT_sent_model

    model_path = os.getenv("AZUREML_MODEL_DIR")

    # Initialize Models
    try:
        LOGGER.info("try attempt...")
        
        #BERT_model_path = os.path.join(model_path, "sent-bert-news", "3", 'SentBERTmodel.pkl')
        #LOGGER.info("MODEL PATH STRING: BERT try 1...")
        #LOGGER.info(BERT_model_path)
        
        BERT_model_path = os.path.join(model_path, 'SentBERTmodel.pkl')
        LOGGER.info("MODEL PATH STRING: BERT try 2...")
        LOGGER.info(BERT_model_path)

        #BERT_model_path = os.path.join(model_path, "sent-bert-news", "3", 'SentBERTmodel.pkl')
        #LOGGER.info("MODEL PATH STRING: BERT try 3...")
        #LOGGER.info(BERT_model_path)

    except:
        BERT_model_path = "../model/SentBERTmodel.pkl"
        
    LOGGER.info("MODEL PATH STRING: BERT...")
    LOGGER.info("MODEL PATH STRING: BERT...")
    LOGGER.info(BERT_model_path)
    LOGGER.info("Filler text...")
    LOGGER.info("Filler text...")


    # Initialize the Models
    # Sentence BERT model
    with open(BERT_model_path, "rb") as f:
        BERT_sent_model = load(f)


def data_collector(input_json_data):
    """Collects data from the input JSON, performs preprocessing and format them into a batch

    Inputs:
        json_payload (dict): JSON input
        
    Outputs:
        lists: returns two lists - filename and text
    """
    import json

    json_payload = json.loads(input_json_data)[0]

    LOGGER.error("JSON Payload inside Data_Collector...")
    LOGGER.error(json_payload)

    article_titles = []
    article_summaries = []
    article_dates = []
    article_urls = []
    article_embeddings = []
    
    # ========== #
    # Parse Raw Data  
    # ========== #

    query = json_payload['modelInput'][0]['query']
    num_results = json_payload['modelInput'][0]['num_results']

    for i in range(len(json_payload['modelInput'][0]['tableData'])):
        article_titles.append(json_payload['modelInput'][0]['tableData'][i]['n_title'])
        article_summaries.append(json_payload['modelInput'][0]['tableData'][i]['n_summary'])
        article_dates.append(json_payload['modelInput'][0]['tableData'][i]['n_date_published'])
        article_urls.append(json_payload['modelInput'][0]['tableData'][i]['n_link'])
        article_embeddings.append(json_payload['modelInput'][0]['tableData'][i]['embeddings'])

    return query, num_results, article_titles, article_summaries, article_dates, article_urls, article_embeddings


def run(input_json_data):

    LOGGER.error("INPUT JSON DATA...")
    LOGGER.error(input_json_data)

    # ========== #
    # Collect Data
    # ========== #
    query, num_results, article_title, article_summary, article_dates, article_urls, article_embeddings = data_collector(input_json_data)

    df = pd.DataFrame()
    df['Article Title'] = article_title
    df['Publish Date'] = article_dates
    df['Article URL'] = article_urls
    df['Article Text'] = article_summary
    
    # ========== #
    # Create Embeddings on Query
    # ========== #
    query_embedding = BERT_sent_model.encode(query)

    # ========== #
    # Measure Similarity
    # ========== #
    df['Similarity Score']  = measure_embedding_similarity(query_embedding, article_embeddings)
    
    # ========== #
    # Retrieve Top K Most Similar Results
    # ========== #
    df = get_similar_texts(df, num_results)

    # ========== #
    # Final Dataset Cleanup
    # ========== #
    columns_to_keep = ['Article Title', 'Publish Date', 'Article URL', 'Article Text', 'Similarity Score']
    df = df[columns_to_keep]

    # ========== #
    ## RESPONSE BACK TO RC
    # ========== #
    output_id = 0

    if len(df) > 0:
        modelOutput =  [
            {"id": output_id, 
            "modelOutput": [item for item in df.to_dict('records')] },
        ]
        return modelOutput
    else:
        return [
            {"id": output_id,
            "modelOutput": [{"Article Title":None, "Publish Date":None, "Article URL": None, "Article Text": None, "Similarity Score": None}]
            },
        ]
        