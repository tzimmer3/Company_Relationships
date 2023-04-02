from src.helpers.nessie import Nessie

import jsonpickle
import pathlib

from src import get_num_tokens, get_tokens

# ================== #
#  Get Number of Tokens
# ================== #

def get_num_tokens(text_2_encode: str, **kwargs):
    """
    Count the number of tokens in a string.
    """
    return len(get_tokens(text_2_encode=text_2_encode, **kwargs))


# ================== #
#  Break Text Into Chunks
# ================== #

# Break text into word / sentence chunks
def break_text_into_chunks(text, chunksize=5000):
    wordchunks = []
    for i in range(0, len(text), chunksize):
        wordchunks.append(text[i:i + chunksize])
    return wordchunks


# ================== #
#  Build Chunks From Dict
# ================== #

def build_chunks_from_dict(cracked_dict: dict,
                                   min_tokens: int = 400,
                                   max_tokens: int = 800, **kwargs) -> dict:
    # Enumerate these in to a dictionary and get the number of tokens
    output_dict = dict()
    total_tokens = 0
    for idx, content in cracked_dict['n_summary'].items():
        output_dict[idx] = {
            'content': content,
            'content_num_tokens': get_num_tokens(text_2_encode=content, **kwargs)
        }

        total_tokens += output_dict[idx]['content_num_tokens']

    chunk_dict = {
        'doc_id': cracked_dict.get('doc_id'),
        'doc_uri': cracked_dict.get('doc_uri'),
        'cracked_uri': None,
        'chunks': {
            0: {
                'chunk_num_tokens': 0,
                'pages': {}
            }
        }
    }
    chunk_number = 0

    # update to be content_num_tokens
    for k, v in output_dict.items():
        if max_tokens > (v['content_num_tokens'] + chunk_dict['chunks'][chunk_number]['chunk_num_tokens']) > min_tokens:
            chunk_dict['chunks'][chunk_number]["pages"][k] = v
            chunk_dict['chunks'][chunk_number]['chunk_num_tokens'] += v['content_num_tokens']

            # Advance Chunk
            chunk_number, chunk_dict = advance_chunk(chunk_number=chunk_number, chunk_dict=chunk_dict)

        elif max_tokens < v['content_num_tokens']:
            if len(chunk_dict['chunks'][chunk_number]['pages']) > 0:
                # Advance Chunk
                chunk_number, chunk_dict = advance_chunk(chunk_number=chunk_number, chunk_dict=chunk_dict)

            page_splits = break_up_text_to_chunks(text=v['content'])
            for split in page_splits:
                num_tokens = get_num_tokens(text_2_encode=split)
                chunk_dict['chunks'][chunk_number]["pages"][k] = {'content': split, 'content_num_tokens': num_tokens}
                chunk_dict['chunks'][chunk_number]['chunk_num_tokens'] += num_tokens

                # Advance Chunk
                chunk_number, chunk_dict = advance_chunk(chunk_number=chunk_number, chunk_dict=chunk_dict)


        elif max_tokens < (v['content_num_tokens'] + chunk_dict['chunks'][chunk_number]['chunk_num_tokens']):
            chunk_number, chunk_dict = advance_chunk(chunk_number=chunk_number, chunk_dict=chunk_dict)

            chunk_dict['chunks'][chunk_number]["pages"][k] = v
            chunk_dict['chunks'][chunk_number]['chunk_num_tokens'] += v['content_num_tokens']
        else:
            chunk_dict['chunks'][chunk_number]["pages"][k] = v
            chunk_dict['chunks'][chunk_number]['chunk_num_tokens'] += v['content_num_tokens']

    return chunk_dict


def advance_chunk(chunk_number: int, chunk_dict: dict):
    chunk_number += 1
    chunk_dict['chunks'][chunk_number] = {
        'chunk_num_tokens': 0,
        'pages': {}
    }
    return chunk_number, chunk_dict


MIN_TOKENS = 400
MAX_TOKENS = 800
SOURCE_FOLDER_NAME = "PagedPDFSplitter"
TARGET_FOLDER_NAME = f"token_{MIN_TOKENS}_{MAX_TOKENS}"

chunky_nessie = Nessie(
    source_config=app_config.CRACKED_LOCATION_CONFIG,
    target_config=app_config.CHUNKED_LOCATION_CONFIG,
)

# Get a list of files in that folder path
# Need to make this a nessie method later
file_list = chunky_nessie.source_file_location.get_filelist(
    folder_path=SOURCE_FOLDER_NAME
)

# # Go through the list
for file in file_list:
    retrieved_bytes = chunky_nessie.source_file_location.get_bytes(file_path=file)
    crack_dict = jsonpickle.loads(retrieved_bytes)

    chunk_dict = build_chunks_from_cracked_dict(cracked_dict=crack_dict,
                                                min_tokens=MIN_TOKENS,
                                                max_tokens=MAX_TOKENS)
    chunk_dict["cracked_uri"] = app_config.AZURE_BLOB_ENDPOINT + \
                                chunky_nessie.source_file_location.container_name + \
                                '/' + file.as_posix()

    # print(crack_dict)
    chunky_nessie.push_improvement_to_target(
        obj=chunk_dict,
        improvement_name=TARGET_FOLDER_NAME,
        orig_folder_name=file.relative_to(chunky_nessie.source_file_location.base_folder).parent.as_posix(),
        overwrite=True,
    )
