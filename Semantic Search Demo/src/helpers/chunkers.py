from typing import  Optional,Union
from tiktoken import Encoding

from .generic_helpers import get_pages_in_range, get_num_tokens

# Constants
MIN_CHUNK_SIZE_CHARS = 100  # The minimum size of each text chunk in characters
MIN_CHUNK_LENGTH_TO_EMBED = 5  # Discard chunks shorter than this
MAX_NUM_CHUNKS = 10000  # The maximum number of chunks to generate from a text

def get_chunks_openai(cracked_dict: dict[str,Union[str,dict]], chunk_token_size: Optional[int],tokenizer:Encoding) -> dict[str,Union[str,dict]]:
    """Chunks text per openai's opensources methodology. Takes in a cracked dict and returns a chunked one. 

    Args:
        cracked_dict (dict[str,Union[str,dict]]): cracked dict where text is stored in a "pages" dict
        chunk_token_size (Optional[int]): desired token size
        tokenizer (Encoding): the tokenizer to use for chunking

    Returns:
        dict[str,Union[str,dict]]: chunked_dict with chunks + associated pages.
    """

    chunk_dict = {
            'doc_id': cracked_dict.get('doc_id'),
            'doc_uri': cracked_dict.get('doc_uri'),
            'cracked_uri': None,
            'chunks': {}
        }

    output_dict = {}
    token_num_dict = {}
    text = ""
    tokens = []
    total_token_num = 0
    for idx, content in cracked_dict['n_summary'].items():
        size = get_num_tokens(content,tokenizer=tokenizer)
        token_num_dict[(total_token_num,total_token_num+size)] = int(idx)
        total_token_num += size
        text += content

    # Return an empty list if the text is empty or whitespace
    if not text or text.isspace():
        return chunk_dict

    # Tokenize the text
    tokens = tokenizer.encode(text, disallowed_special=())

    num_chunks = 0

    # Loop until all tokens are consumed
    while tokens and num_chunks < MAX_NUM_CHUNKS:
        # Take the first chunk_size tokens as a chunk
        chunk = tokens[:chunk_token_size]

        # Decode the chunk into text
        chunk_text = tokenizer.decode(chunk)

        # Skip the chunk if it is empty or whitespace
        if not chunk_text or chunk_text.isspace():
            # Remove the tokens corresponding to the chunk text from the remaining tokens
            tokens = tokens[len(chunk) :]
            # Continue to the next iteration of the loop
            continue

        # Find the last period or punctuation mark in the chunk
        last_punctuation = max(
            chunk_text.rfind("."),
            chunk_text.rfind("?"),
            chunk_text.rfind("!"),
            chunk_text.rfind("\n"),
        )

        # If there is a punctuation mark, and the last punctuation index is before MIN_CHUNK_SIZE_CHARS
        if last_punctuation != -1 and last_punctuation > MIN_CHUNK_SIZE_CHARS:
            # Truncate the chunk text at the punctuation mark
            chunk_text = chunk_text[: last_punctuation + 1]

        # Remove any newline characters and strip any leading or trailing whitespace
        chunk_text_to_append = chunk_text.replace("\n", " ").strip()
        chunk_token_len = len(tokenizer.encode(chunk_text, disallowed_special=()))

        if len(chunk_text_to_append) > MIN_CHUNK_LENGTH_TO_EMBED:
            output_dict[str(num_chunks)] = {
                "n_summary": get_pages_in_range(token_num_dict,total_token_num-len(tokens),total_token_num-len(tokens)+chunk_token_len),
                "content":chunk_text_to_append
            }

        # Remove the tokens corresponding to the chunk text from the remaining tokens
        tokens = tokens[ chunk_token_len:]

        # Increment the number of chunks
        num_chunks += 1

    # Handle the remaining tokens
    if tokens:
        remaining_text = tokenizer.decode(tokens).replace("\n", " ").strip()
        if len(remaining_text) > MIN_CHUNK_LENGTH_TO_EMBED:
            output_dict[str(num_chunks)] = {
                "n_summary": get_pages_in_range(token_num_dict,total_token_num-len(tokens),total_token_num),
                "content":remaining_text
            }

    chunk_dict["chunks"] = output_dict

    return chunk_dict