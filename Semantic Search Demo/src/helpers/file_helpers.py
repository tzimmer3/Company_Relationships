import pathlib
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, AnyUrl

from .blob_helpers import (
    get_filelist_blob,
    get_blob_client,
    get_container_client,
    get_blob_service_client,
)
from .generic_helpers import (
    get_filelist_local,
    stat_to_json,
    get_datetime_path,
)
from .hashers import get_md5_hash_of_file


class PydanticFile(BaseModel):
    content: str
    url: Optional[AnyUrl]
    source_type: Optional[str]
    filepath: Optional[str]
    fileinfo: Optional[dict]


class FileLocation(object):
    type: str = None
    container_name: str = None
    base_folder: pathlib.Path = None
    connection_string: str = None
    blob_service_client: BlobServiceClient = None
    container_client: ContainerClient = None
    folder_pattern: str = (
        "%Y/%m"  # folder pattern for date based folder structure
    )

    def __init__(self, config: dict = None):
        if config:
            self.configure(config=config)

    @classmethod
    def from_config(cls, config: dict):
        file_loc = FileLocation()

        file_loc.configure(config=config)

        return file_loc

    def configure(self, config: dict):
        """Takes a dict of configs and set attributes on self"""
        for k, v in config.items():
            if type(v) == str:
                if k == "base_folder":
                    setattr(self, k, pathlib.Path(v))
                    # setattr(self, k, v)
                else:
                    setattr(self, k, v)
            elif type(v) == pathlib.Path:
                setattr(self, k, v)

    def path_check(self, file_path=None, folder_name=None, file_name=None):
        assert (
            (folder_name is not None and file_name is not None)
            or file_path is not None
            or file_name is not None
        ), "use your words because I don't know what you want to work with."

        # Create remote file path if needed
        if file_path is None and folder_name is None:
            folder_name = get_datetime_path(
                dt=datetime.utcnow(), path_format=self.folder_pattern
            )
            file_path = self.base_folder.joinpath(folder_name, file_name)
        elif (
            file_path is None
            and folder_name is not None
            and file_name is not None
        ):
            file_path = self.base_folder.joinpath(folder_name, file_name)

        return (
            pathlib.Path(file_path)
            if file_path
            else self.base_folder.joinpath(folder_name, file_name)
        )

    def get_blob_client(
        self,
        folder_name: str = None,
        file_name: str = None,
        file_path: str = None,
    ):
        file_path = self.path_check(
            file_path=file_path, folder_name=folder_name, file_name=file_name
        )

        if self.blob_service_client is None:
            self.blob_service_client = get_blob_service_client(
                connection_string=self.connection_string
            )

        blob_client = get_blob_client(
            blob_service_client=self.blob_service_client,
            container_name=self.container_name,
            file_path=file_path.as_posix(),
        )

        return blob_client

    def get_blob_container_client(self):
        if self.blob_service_client is None:
            self.blob_service_client = get_blob_service_client(
                self.connection_string
            )

        if self.container_client is None:
            self.container_client = get_container_client(
                blob_service_client=self.blob_service_client,
                container_name=self.container_name,
            )

        return self.container_client

    def get_file(
        self,
        folder_name: str = None,
        file_name: str = None,
        file_path: str = None,
    ):
        """
        gets file contents and info

        folder_name: str = This is the folder
        file_name: str = This is the filename
        file_path: str = This is the full path to the file including filename and folder
        You must provide either the folder and file or the file path
        """
        file_path = self.path_check(
            file_path=file_path, folder_name=folder_name, file_name=file_name
        )

        # Set export variable
        output = None

        if self.type == "local":
            if file_path.exists():
                file_stats = file_path.stat()

                with file_path.open(mode="rt") as f:
                    file_text = f.read()

                output = PydanticFile(
                    content=file_text,
                    fileinfo=stat_to_json(file_stats),
                    source_type=self.type,
                    filepath=file_path.as_posix(),
                )

        elif self.type == "blob":
            blob_client = self.get_blob_client(file_path=file_path.as_posix())
            output = None

            if blob_client.exists():
                file_stats = blob_client.get_blob_properties()
                file_text = blob_client.download_blob().readall()

                output = PydanticFile(
                    content=file_text.decode(encoding="utf-8"),
                    fileinfo=stat_to_json(file_stats),
                    url=blob_client.url,
                    source_type=self.type,
                    filepath=file_path.as_posix(),
                )

        return output

    def put_file(
        self,
        text: str,
        overwrite: bool = True,
        folder_name: str = None,
        file_name: str = None,
        file_path: str = None,
    ):
        """Puts the text string in to a file"""
        status = None

        file_path = self.path_check(
            file_path=file_path, folder_name=folder_name, file_name=file_name
        )

        if self.type == "blob":
            # make blob service client
            blob_client = self.get_blob_client(file_path=file_path.as_posix())
            if blob_client.exists() and overwrite is False:
                print(
                    "WARNING: blob file exists and overwrite was set to false"
                )
                return False

            blob_client.upload_blob(text, overwrite=True)

            return True

        elif self.type == "local":
            if file_path.exists() and overwrite is False:
                print(
                    "WARNING: local file exists and overwrite was set to false"
                )
                return False

            # Make the directory if it doesn't exist
            file_path.parent.mkdir(parents=True, exist_ok=True)

            with file_path.open(mode="wt") as f:
                f.write(text)

            return True

        return status

    def put_bytes(
        self,
        bytes,
        overwrite: bool = True,
        folder_name: str = None,
        file_name: str = None,
        file_path: str = None,
    ):
        """Puts the bytes in to a file"""
        status = None

        file_path = self.path_check(
            file_path=file_path, folder_name=folder_name, file_name=file_name
        )

        if self.type == "blob":
            # make blob service client
            blob_client = self.get_blob_client(file_path=file_path.as_posix())
            if blob_client.exists() and overwrite is False:
                print(
                    "WARNING: blob file exists and overwrite was set to false"
                )
                return False

            blob_client.upload_blob(bytes, overwrite=True)

            return True

        elif self.type == "local":
            if file_path.exists() and overwrite is False:
                print(
                    "WARNING: local file exists and overwrite was set to false"
                )
                return False

            # Make the directory if it doesn't exist
            file_path.parent.mkdir(parents=True, exist_ok=True)

            with file_path.open(mode="wb") as f:
                f.write(bytes)

            return True

        return status

    def get_bytes(
        self,
        folder_name: str = None,
        file_name: str = None,
        file_path: str = None,
    ):
        file_path = self.path_check(
            file_path=file_path, folder_name=folder_name, file_name=file_name
        )

        # Set export variable
        output = None

        if self.type == "local":
            if file_path.exists():
                with file_path.open(mode="rb") as f:
                    output = f.read()

        elif self.type == "blob":
            blob_client = self.get_blob_client(file_path=file_path.as_posix())
            if blob_client:
                if blob_client.exists():
                    output = blob_client.download_blob().readall()

        return output

    def get_endpoint(
        self,
        folder_name: str = None,
        file_name: str = None,
        file_path: str = None,
    ):
        file_path = self.path_check(
            file_path=file_path, folder_name=folder_name, file_name=file_name
        )

        # Set export variable
        output = None

        if self.type == "local":
            if file_path.exists():
                output = file_path.as_posix()

        elif self.type == "blob":
            blob_client = self.get_blob_client(file_path=file_path.as_posix())
            if blob_client:
                if blob_client.exists():
                    output = blob_client.primary_endpoint

        return output

    def delete_file(
        self,
        file_path: str = None,
        folder_name: str = None,
        file_name: str = None,
    ):
        file_path = self.path_check(
            file_path=file_path, folder_name=folder_name, file_name=file_name
        )

        output = None

        if self.type == "local":
            if file_path.exists():
                file_path.unlink(missing_ok=False)
                output = True
            else:
                output = False

        elif self.type == "blob":
            blob_client = self.get_blob_client(file_path=file_path.as_posix())

            if blob_client.exists():
                blob_client.delete_blob(delete_snapshots="include")
                output = True
            else:
                output = False

        return output

    def get_filelist(
        self,
        root_path: pathlib.Path = None,
        folder_path: pathlib.Path = None,
        file_types: list = None,
        oldest_datetime: datetime = None,
        newest_datetime: datetime = None,
    ) -> list[pathlib.Path]:
        # Create remote file path if needed
        if folder_path is not None and root_path is None:
            search_path = self.base_folder.joinpath(folder_path)
        elif folder_path is None and root_path is not None:
            search_path = root_path
        else:
            search_path = self.base_folder

        if self.type == "local":
            return get_filelist_local(
                root_path=search_path,
                file_types=file_types,
                oldest_datetime=oldest_datetime,
                newest_datetime=newest_datetime,
            )

        elif self.type == "blob":
            container_client = self.get_blob_container_client()

            return get_filelist_blob(
                root_path=search_path,
                file_types=file_types,
                oldest_datetime=oldest_datetime,
                newest_datetime=newest_datetime,
                container_client=container_client,
            )

    @classmethod
    def from_dict(cls, configuration: dict):
        file_location = FileLocation()

        for k, v in configuration.items():
            if k == "type":
                file_location.type = v
            elif k == "container_name":
                file_location.container_name = v
            elif k == "base_folder":
                file_location.base_folder = pathlib.Path(v)
            elif k == "connection_string":
                file_location.connection_string = v
            elif k == "folder_pattern":
                file_location.folder_pattern = v

        file_location.load_config()

        return file_location

    @classmethod
    def from_config(cls):
        file_location = FileLocation()

        file_location.load_config()

        return file_location

    def load_config(self):
        if not self.type:
            self.type = "blob"

    def get_file_properties(
        self,
        file_path: str = None,
        folder_name: str = None,
        file_name: str = None,
    ):
        file_path = self.path_check(
            file_path=file_path, folder_name=folder_name, file_name=file_name
        )

        file_md5 = None
        file_size = None
        file_properties = None
        if self.type == "local":
            if file_path.exists():
                file_md5 = get_md5_hash_of_file(filepath=file_path)
                file_properties = file_path.lstat()
                file_size = file_properties.st_size
            else:
                return None
        elif self.type == "blob":
            blobclient = self.get_blob_client(
                folder_name=folder_name, file_name=file_name
            )
            if blobclient.exists():
                file_properties = blobclient.get_blob_properties()
                file_size = file_properties.size
                file_md5 = file_properties.content_settings.content_md5.hex()
            else:
                return None

        return {
            "file_path": file_path.as_posix(),
            "md5": file_md5,
            "size": file_size,
            "properties": file_properties,
        }

    def get_file_bytes_by_uri(self,file_uri:str):
        container_client = self.get_blob_container_client()
        file_path = file_uri.removeprefix(container_client.primary_endpoint).removeprefix("/")
        return self.get_bytes(file_path=file_path)