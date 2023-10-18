import os
from pathlib import Path
from typing import Union

def mkdir_if_not_exists(p):
    if not os.path.isdir(p):
        print(f"dir `{p}` does not exist, create now")
        os.makedirs(p)

def get_n_last_subparts_path(base_dir: Union[Path, str], n:int) -> Path:
    return Path(*Path(base_dir).parts[-n-1:])