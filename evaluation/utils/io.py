import json
from typing import Dict


def save_json(content: Dict, path: str, indent: int = 4, **kwargs) -> None:
    with open(path, "w") as f:
        json.dump(content, f, indent=indent, sort_keys=True, **kwargs)

def load_json(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)
