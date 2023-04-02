#from .helpers.file_helpers import FileLocation

import hashlib
import jsonpickle

from pathlib import Path




# Get md5 hash of a file
def get_md5_hash_of_file(filepath):
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()



class Nessie(object):
    """
    Class to facilitate data improvements on files in a lake using md5 hashes for deconflicting source files
    """

    def __init__(self, source_config: dict, target_config: dict):
        self.source_file_location: FileLocation = FileLocation.from_dict(
            source_config
        )
        self.target_file_location: FileLocation = FileLocation.from_dict(
            target_config
        )

    def get_file_locally(
        self,
        folder_name: str = None,
        file_name: str = None,
        file_path: str = None,
        source_file_location: FileLocation = None,
        local_file_location: FileLocation = None,
        overwrite_local: bool = True,
        **kwargs,
    ):
        if source_file_location is None:
            source_file_location = self.source_file_location

        assert (
            local_file_location is not None
        ), "must provide local file location"

        file_bytes = source_file_location.get_bytes(
            folder_name=folder_name, file_name=file_name, file_path=file_path
        )
        if file_bytes:
            print(f"{folder_name}/{file_name} - pulled")
        else:
            print(f"{folder_name}/{file_name} - pull failed")
            return None, None

        if local_file_location.put_bytes(
            bytes=file_bytes,
            overwrite=overwrite_local,
            file_name=file_name,
            folder_name=folder_name,
            file_path=file_path,
        ):
            print(f"{folder_name}/{file_name} - written locally")
        else:
            print(f"{folder_name}/{file_name} - local write failed")
            return None, None

        local_location_path = local_file_location.path_check(
            file_name=file_name, folder_name=folder_name
        )

        return get_md5_hash_of_file(local_location_path), local_location_path

    def push_file_to_target(
        self,
        improvement_folder: str,
        overwrite: bool = False,
        local_file_path: Path = None,
        local_file_name: str = None,
        local_folder_name: str = None,
        local_file_location: FileLocation = None,
        target_file_location: FileLocation = None,
    ):
        assert local_file_path is not None or (
            local_file_name is not None
            and local_file_location is not None
            and local_folder_name is not None
        ), "You gotta gimme me a way to find this file. Either file_path or (file_name, folder_name and FileLocation)"

        if local_file_location is None:
            local_file_location = self.local_file_location

        if target_file_location is None:
            target_file_location = self.target_file_location

        if local_folder_name is None:
            local_folder_name = local_file_path.relative_to(
                local_file_location.base_folder
            ).parent

        if local_file_name is None:
            local_file_name = local_file_path.relative_to(
                local_file_location.base_folder
            ).name

        if local_file_path is None:
            local_file_path = local_file_location.path_check(
                folder_name=local_folder_name, file_name=local_file_name
            )

        target_folder = Path(improvement_folder).joinpath(local_folder_name)

        file_bytes = local_file_path.read_bytes()

        if target_file_location.put_bytes(
            bytes=file_bytes,
            file_name=local_file_name,
            folder_name=target_folder.as_posix(),
            overwrite=overwrite,
        ):
            print(
                f"{target_folder.as_posix()}/{local_file_name} - written at target"
            )
            return True
        else:
            print(
                f"{target_folder.as_posix()}/{local_file_name} - target write failed"
            )
            return None

    def push_object_to_target_as_json(
        self,
        obj,
        target_file_name: str,
        target_folder_name: str,
        overwrite: bool = False,
        target_file_location: FileLocation = None,
    ):
        """Takes an object as a json using jsonpickle to the file location."""
        if target_file_location is None:
            target_file_location = self.target_file_location

        if target_file_location.put_file(
            text=jsonpickle.encode(obj, unpicklable=False, indent=2),
            file_name=target_file_name,
            folder_name=target_folder_name,
            overwrite=overwrite,
        ):
            print(
                f"{target_folder_name}/{target_file_name} - written at target"
            )
            return True
        else:
            print(
                f"{target_folder_name}/{target_file_name} - target write failed"
            )
            return None

    def push_improvement_to_target(
        self,
        obj,
        improvement_name: str,
        doc_id: str = None,
        doc_uri: str = None,
        orig_folder_name: str = None,
        target_file_location: FileLocation = None,
        overwrite: bool = False,
        **kwargs,
    ):
        if orig_folder_name:
            target_folder_name = f"{improvement_name}/{orig_folder_name}"
        else:
            target_folder_name = improvement_name

        if type(obj) == dict and obj.get("doc_id") != None:
            output = obj
            doc_id = obj.get("doc_id")
        else:
            output = {
                "doc_id": doc_id,
                "doc_uri": doc_uri,
                improvement_name: obj,
            }

        return self.push_object_to_target_as_json(
            obj=output,
            target_file_name=f"{doc_id}.json",
            target_folder_name=target_folder_name,
            target_file_location=target_file_location,
            overwrite=overwrite,
        )


# src_file = lake_path.joinpath("raw/arxiv/Retrieval_Augmented_Generation/2104.07713v2.pdf")

if __name__ == "__main__":
    nessie = Nessie()
    doc_id, local_test_file_path = nessie.get_file_locally(
        file_name="2104.07713v2.pdf",
        folder_name="arxiv/Retrieval_Augmented_Generation",
        overwrite_local=True,
    )

    nessie.push_file_to_target(
        improvement_folder="test/improvement",
        local_file_path=local_test_file_path,
    )
