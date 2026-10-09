"""Character IDs and windows made within already-split fictional sequences."""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

DATA_PATH = Path(__file__).with_name("sequences.json")


def load_data(path=DATA_PATH):
    records = json.loads(path.read_text())
    train = [row["text"] for row in records if row["split"] == "train"]
    validation = [row["text"] for row in records if row["split"] == "validation"]
    if not train or not validation:
        raise ValueError("Both training and validation sequences are required.")
    if set(train) & set(validation):
        raise ValueError("A complete sequence appears in both splits.")
    vocabulary = sorted(set("".join(train)))
    unknown = set("".join(validation)) - set(vocabulary)
    if unknown:
        raise ValueError(f"Validation contains unknown characters: {unknown!r}")
    return train, validation, vocabulary


def encode(text, vocabulary):
    char_to_id = {char: index for index, char in enumerate(vocabulary)}
    return [char_to_id[char] for char in text]


def decode(ids, vocabulary):
    return "".join(vocabulary[index] for index in ids)


class CharacterWindows(Dataset):
    """Stride-one windows; a window never crosses a sequence boundary."""

    def __init__(self, texts, vocabulary, context_length):
        if context_length < 1:
            raise ValueError("Context length must be positive.")
        self.windows = []
        for text in texts:
            ids = encode(text, vocabulary)
            for start in range(len(ids) - context_length):
                window = torch.tensor(
                    ids[start : start + context_length + 1], dtype=torch.long
                )
                self.windows.append((window[:-1], window[1:]))
        if not self.windows:
            raise ValueError("No sequences are longer than the context length.")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        return self.windows[index]
