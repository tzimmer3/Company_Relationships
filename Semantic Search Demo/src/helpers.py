import tiktoken



# ================== #
# Tokenize Text
# ================== #

# TODO: change tiktoken out to real tokenizer... or use tiktoken

def get_tokens(text_2_encode: str, tokenizer=None):
    """
    Tokenize text in a string.

    Initialize a tokenizer if tokenizer == None.
    """

    if tokenizer is None:
        tokenizer = tiktoken.encoding_for_model("text-davinci-003")
    return tokenizer.encode(text=text_2_encode)




def get_num_tokens(text_2_encode: str, **kwargs):
    """
    Count the number of tokens in a string.
    """
    return len(get_tokens(text_2_encode=text_2_encode, **kwargs))




# ================== #
# Get Pages in Range
# ================== #

def get_pages_in_range(my_dict:dict, min_token:int, max_token:int) -> list[str]:
    
    
    """
    Takes in a dict whose keys are a tuple representing the min and max token and
    whose value is the page that that range corresponds to. Returns all pages that 
    overlap with this range. Used for the openai chunking method.

    Args:
        my_dict (dict): dict of token range : pages
        min_token (int): min_token for current chunk
        max_token (int): max_token for current chunk

    Returns:
        list: list of strings representing pages in that range.

    """
    keys = []
    for (key_min,key_max), val in my_dict.items():
        if (key_min <= min_token <= key_max) or (key_min <= max_token <= key_max):
            keys.append(val)
    return keys



"""

def get_chunk_content_from_pages(pages:dict[str, dict[str,Union[int,str]]]) -> str:
    content = ""
    for page in pages:
        content += pages[page]["content"]

    return content

def get_tokens_from_text(text_2_encode: str, tokenizer=None) -> list:
    if tokenizer is None:
        tokenizer = tiktoken.encoding_for_model("text-davinci-003")
    return tokenizer.encode(text=text_2_encode)


def get_text_from_tokens(tokens_2_decode: list, tokenizer=None) -> str:
    if tokenizer is None:
        tokenizer = tiktoken.encoding_for_model("text-davinci-003")
    return tokenizer.decode(tokens=tokens_2_decode)


def break_up_text_to_chunks(text, max_tokens=800, overlap_pct: float = .25):
    tokens = get_tokens_from_text(text_2_encode=text)
    num_tokens = len(tokens)
    chunk_count = math.ceil(num_tokens/max_tokens)
    chunk_size = math.ceil(num_tokens/chunk_count)
    overlap_size = math.ceil((max_tokens - chunk_size) * overlap_pct)
    output_tokens = list(break_up_list_of_tokens(tokens, chunk_size + overlap_size, overlap_size))
    return [get_text_from_tokens(tokens_2_decode=token_list) for token_list in output_tokens]



# TODO: add some jsonpickle to the files to reduce file size

#file_path = pathlib.Path(r"C:\Users\natha\AppData\Local\Temp\16795094427753\5c7787373afaac2c553dded3ba29a82e.json")
#file_path = pathlib.Path(r"C:\Users\natha\AppData\Local\Temp\16795137134516\5c7787373afaac2c553dded3ba29a82e.json")

#doc = jsonpickle.decode(file_path.read_text())


"""