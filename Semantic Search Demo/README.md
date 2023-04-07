## Create a Semantic Search Engine from Exxon articles.

Goal: Demonstrate how to build a search engine using a Large Language Model.  Enable a Question/Answer style of interaction with ~1000 news articles about Exxon.


Note: Most of the scripts in this folder are not working yet.  Working code is in the DEMO notebooks (Search Demo and Search Demo Chunked)


TODO List:
    - Test out different chunking methodologies
    - Test out other embedding strategies (other LLMs)
    - [Project Specific] Store each article as its own JSON file (instead of one big file)
    - Add in Lake functionality... stop working with local files

### Sentence BERT Model

sentence-transformers/all-MiniLM-L6-v2

By default, input text longer than 256 word pieces is truncated.



### Chunked Documents Schema
 - Article Title
 - Article Content
 - Article Publish Date
 - Article URL
 - Chunk Number