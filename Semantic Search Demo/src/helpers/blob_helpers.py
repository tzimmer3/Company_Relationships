# Import from helper package
from . import *

from .generic_helpers import is_datetime_path_between
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import os
import pathlib
from datetime import datetime

MAX_SINLGE_PUT_SIZE = 16*1024*1024

def get_blob_client(
    container_name: str,
    file_path: str,
    blob_service_client: BlobServiceClient = None,
    connection_string: str = None,
) -> BlobClient:
    if not blob_service_client:
        blob_service_client = get_blob_service_client(
            connection_string=connection_string
        )

    # Create a blob client
    blob_client = blob_service_client.get_blob_client(
        container=container_name, blob=file_path
    )
    

    return blob_client


def get_blob_service_client_with_connection_string(
    connection_string: str = None, **kwargs
):
    if not connection_string:
        connection_string = kwargs.get("AZURE_STORAGE_CONNECTION_STRING")

    return BlobServiceClient.from_connection_string(connection_string,max_single_put_size=MAX_SINLGE_PUT_SIZE)


def get_blob_service_client_with_service_principal(
    azure_tenant_id: str = None,
    azure_application_id: str = None,
    azure_application_secret: str = None,
    azure_blob_endpoint: str = None,
    **kwargs
):
    # azure_tenant_id = azure_tenant_id if azure_tenant_id else app_config.AZURE_TENANT_ID
    # azure_application_id = azure_application_id if azure_application_id else app_config.AZURE_CLIENT_ID
    # azure_application_secret = azure_application_secret if azure_application_secret else app_config.AZURE_CLIENT_SECRET
    # azure_blob_endpoint = azure_blob_endpoint if azure_blob_endpoint else app_config.AZURE_BLOB_ENDPOINT

    assert (
        azure_tenant_id and azure_application_id and azure_application_secret
    ), "need azure sp creds"

    from azure.identity import ClientSecretCredential

    token_credential = ClientSecretCredential(
        azure_tenant_id, azure_application_id, azure_application_secret
    )

    # Instantiate a BlobServiceClient using a token credential
    from azure.storage.blob import BlobServiceClient

    blob_service_client = BlobServiceClient(
        account_url=azure_blob_endpoint, credential=token_credential,max_single_put_size=MAX_SINLGE_PUT_SIZE
    )

    return blob_service_client


def get_blob_service_client_with_default_azure_cred(
    azure_blob_endpoint: str = None, **kwargs
):
    # azure_blob_endpoint = azure_blob_endpoint if azure_blob_endpoint else app_config.AZURE_BLOB_ENDPOINT

    from azure.identity import DefaultAzureCredential
    from azure.storage.blob import BlobServiceClient

    default_credential = DefaultAzureCredential()

    blob_service_client = BlobServiceClient(
        account_url=azure_blob_endpoint, credential=default_credential,max_single_put_size=MAX_SINLGE_PUT_SIZE
    )

    return blob_service_client


def get_blob_service_client(
    connection_string: str = None,
    azure_tenant_id: str = None,
    azure_application_id: str = None,
    azure_application_secret: str = None,
    azure_blob_endpoint: str = None,
):
    if connection_string:
        blob_service_client = get_blob_service_client_with_connection_string(
            connection_string=connection_string
        )
    elif (
        azure_application_id
        and azure_application_secret
        and azure_blob_endpoint
    ):
        blob_service_client = get_blob_service_client_with_service_principal(
            azure_tenant_id=azure_tenant_id,
            azure_application_id=azure_application_id,
            azure_application_secret=azure_application_secret,
            azure_blob_endpoint=azure_blob_endpoint,
        )
    else:
        blob_service_client = get_blob_service_client_with_default_azure_cred(
            azure_blob_endpoint=azure_blob_endpoint
        )

    return blob_service_client


def get_container_client(
    container_name: str,
    blob_service_client: BlobServiceClient = None,
    connection_string: str = None,
):
    if not blob_service_client:
        blob_service_client = get_blob_service_client(
            connection_string=connection_string
        )

    return blob_service_client.get_container_client(container=container_name)


def upload_text_to_blob(
    container_name: str,
    remote_folder_path: str,
    file_name: str,
    text: str,
    overwrite: bool = False,
):
    """
    uploads a string to a blob

    args:
        container_name = name of the container
        remote_folder_path = the path in the container
        file_name = the name of the new file
        text = the text to insert in the file
    return:
        If it uploaded.
    """

    # Create remote file path
    remote_file_path = os.path.join(remote_folder_path, file_name)

    # make blob service client
    blob_client = get_blob_client(
        container_name=container_name, file_path=remote_file_path
    )

    if blob_client.exists() and not overwrite:
        blob_properties = blob_client.get_blob_properties()
        if blob_properties.size != len(text):
            raise Exception("Uh Oh, same md5 and different size!")
        return False
    else:
        print("--- Uploading to Azure Storage as blob:\t" + file_name)
        blob_client.upload_blob(text, overwrite=overwrite)
        return True


def download_text_from_blob(
    container_name: str,
    remote_folder_path: str = None,
    file_name: str = None,
    remote_file_path: str = None,
):
    """
    downloads a string to a blob

    args:
        container_name = name of the container
        remote_folder_path = the path in the container
        file_name = the name of the new file

    return:
        text
    """
    if remote_file_path is None:
        remote_file_path = os.path.join(remote_folder_path, file_name)

    # make blob service client
    blob_client = get_blob_client(
        container_name=container_name, file_path=remote_file_path
    )

    if blob_client.exists():
        text = blob_client.download_blob().readall()

        return text

    else:
        return None


def upload_file_to_blob(
    container_name: str,
    remote_folder_path: str,
    local_folder_path: str,
    file_name: str,
):
    """
    uploads a file to a blob

    args:
        container_name = name of the container
        remote_folder_path = the path in the container
        local_folder_path = where to put the file
        file_name = the name of the file

    return:
        success: bool
    """
    local_file_path = os.path.join(local_folder_path, file_name)
    remote_file_path = os.path.join(remote_folder_path, file_name)

    # Create a blob client
    blob_client = get_blob_client(
        container_name=container_name, file_path=remote_file_path
    )

    if blob_client.exists():
        blob_properties = blob_client.get_blob_properties()
        if blob_properties.size != os.stat(local_file_path).st_size:
            raise Exception("Uh Oh, same md5 and different size!")
        return False
    else:
        print("--- Uploading to Azure Storage as blob:\t" + file_name)
        with open(local_file_path, "rb") as data:
            blob_client.upload_blob(data)
        return True


def download_file_from_blob(
    container_name: str,
    remote_folder_path: str,
    local_folder_path: str,
    file_name: str,
):
    """
    downloads a file from a blob

    args:
        container_name = name of the container
        remote_folder_path = the path in the container
        local_folder_path = where to put the file
        file_name = the name of the file

    return:
        success: bool
    """
    local_file_path = os.path.join(local_folder_path, file_name)
    remote_file_path = os.path.join(remote_folder_path, file_name)

    # Create a blob client
    blob_client = get_blob_client(
        container_name=container_name, file_path=remote_file_path
    )

    if blob_client.exists():
        with open(local_file_path, "wb") as file:
            file.write(blob_client.download_blob().readall())

        return True
    else:
        return False


def get_filelist_blob(
    root_path: pathlib.Path = None,
    file_types: list = None,
    oldest_datetime: datetime = None,
    newest_datetime: datetime = None,
    container_name: str = None,
    blob_service_client: BlobServiceClient = None,
    container_client: ContainerClient = None,
    connection_string: str = None,
) -> list[pathlib.Path]:
    """
    returns a list of files recursively from newer directories

    Assumes directories are in yyyy/mm or yyyy/mm/dd format
    root_path: Optional[pathlib.Path] = at what path to begin search
    file_types: Optional[list] = List of file extentions to return ie ['.html', '.pdf']
    oldest_date: Optional[datetime.date] = gets all files of the same date or newer

    """

    # Convert to pathlib.Path
    root_path = pathlib.Path(root_path)

    # Get container client
    if container_client is None:
        cclient = get_container_client(
            container_name=container_name,
            blob_service_client=blob_service_client,
            connection_string=connection_string,
        )
    else:
        cclient = container_client

    # Build the file_list
    file_list = list()
    for blob in cclient.list_blobs(name_starts_with=root_path.as_posix()):
        # Convert to pathlib.Path if not already and get common relpath
        current_path = pathlib.Path(blob.name)
        # print(f"{current_path}")

        # If dates are provided, verify current directory is in range otherwise skip directory
        if (
            oldest_datetime is not None or newest_datetime is not None
        ) and not is_datetime_path_between(
            current_path=current_path,
            oldest_datetime=oldest_datetime,
            newest_datetime=newest_datetime,
            root_path=root_path,
        ):
            # print("skipped")
            continue

        if file_types is None or current_path.suffix in file_types:
            file_list.append(current_path)

    return file_list
