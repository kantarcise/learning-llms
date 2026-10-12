"""Character IDs and windows made within already-split fictional sequences."""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

DATA_PATH = Path(__file__).with_name("sequences.json")


def load_data(path=DATA_PATH):
    """Read the hand-written corpus and construct our character tokenizer's vocabulary.

    Args:
        path: JSON file containing records with ``text`` and ``split`` fields.
            This function reads existing text; it does not generate log data.

    Returns:
        (train_texts, validation_texts, vocabulary), three lists. Vocabulary is
        a sorted list of distinct Unicode characters in training text. A
        character's list index is its token ID; these are not UTF-8 byte IDs.

    Raises:
        ValueError: A split is empty, a whole record occurs in both splits, or
            validation contains a character absent from the training vocabulary.

    This deliberately minimal tokenizer has no unknown-token fallback or special
    end token. Reuse the same vocabulary for validation and generation. Changing
    its contents or ordering changes token IDs and invalidates existing weights.
    Whole-record separation does not eliminate shared templates or substrings.
    """
    records = json.loads(path.read_text())
    train = [row["text"] for row in records if row["split"] == "train"]
    validation = [row["text"] for row in records if row["split"] == "validation"]
    if not train or not validation:
        raise ValueError("Both training and validation sequences are required.")
    if set(train) & set(validation):
        raise ValueError("A complete sequence appears in both splits.")
    # Fit the vocabulary on training text only; validation must not expand it.
    vocabulary = sorted(set("".join(train)))
    unknown = set("".join(validation)) - set(vocabulary)
    if unknown:
        raise ValueError(f"Validation contains unknown characters: {unknown!r}")
    return train, validation, vocabulary


def encode(text, vocabulary):
    """Map each Unicode character to its index in the supplied vocabulary.

    Returns a list of integer IDs, one per Python string character. For example,
    encode("cab", ["a", "b", "c"]) returns [2, 0, 1]. An unseen character raises
    KeyError. The caller must reuse the vocabulary associated with the model.
    """
    char_to_id = {char: index for index, char in enumerate(vocabulary)}
    return [char_to_id[char] for char in text]


def decode(ids, vocabulary):
    """Join vocabulary entries for a sequence of valid token IDs into text.

    This reverses encode when both use the same vocabulary. IDs must be in
    [0, len(vocabulary)); model predictions already satisfy that constraint.
    """
    return "".join(vocabulary[index] for index in ids)


class CharacterWindows(Dataset):
    """Prepare overlapping input/target pairs for PyTorch's DataLoader.

    Args:
        texts: Already-split source strings (training OR validation).
        vocabulary: Ordered character list shared by the tokenizer and model.
        context_length: Number T of input positions in each example.

    Each item is two torch.long tensors of shape [T]. A slice of T+1 IDs
    produces input=slice[:-1] and target=slice[1:]. DataLoader stacks these
    into [B,T] batches. Windows advance by one character and never cross records.
    This small example materializes all windows in memory; it is not a streaming
    loader for large corpora. Overlapping windows are not independent records.
    """

    def __init__(self, texts, vocabulary, context_length):
        if context_length < 1:
            raise ValueError("Context length must be positive.")
        self.windows = []
        # Each record belongs to a split BEFORE we create overlapping windows.
        # This prevents adjacent windows from one record landing in both splits.
        for text in texts:
            ids = encode(text, vocabulary)
            for start in range(len(ids) - context_length):
                window = torch.tensor(
                    ids[start : start + context_length + 1], dtype=torch.long
                )
                # Take T+1 characters: the last T are the next-character labels.
                # Example: job=orde -> ob=order (the source slice has 9 chars).
                self.windows.append((window[:-1], window[1:]))
        if not self.windows:
            raise ValueError("No sequences are longer than the context length.")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        return self.windows[index]
