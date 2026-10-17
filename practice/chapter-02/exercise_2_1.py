"""Inspect GPT-2 token pieces for Exercise 2.1 of Raschka's book.

Run from the repository root:
    uv run --locked --group book python practice/chapter-02/exercise_2_1.py

The first run may download GPT-2's tokenizer data into tiktoken's cache.
No model weights are loaded or trained. The selected encoding determines IDs.
"""

from importlib.metadata import version

import tiktoken


def main() -> None:
    """Print the exact input, token IDs, byte pieces, and decoded round trip."""
    tokenizer = tiktoken.get_encoding("gpt2")
    text = "Akwirw ier"
    expected_ids = [33901, 86, 343, 86, 220, 959]
    ids = tokenizer.encode(text)

    print(f"tiktoken: {version('tiktoken')}; encoding: {tokenizer.name}")
    print(f"Input: {text!r}")
    print(f"IDs: {ids}")
    print(f"Matches expected IDs: {ids == expected_ids}")
    print("ID -> raw bytes -> decoded piece")
    for token_id in ids:
        # A token can contain only part of a UTF-8 character. Raw bytes are
        # authoritative; all pieces in this particular example are ASCII.
        raw_bytes = tokenizer.decode_single_token_bytes(token_id)
        print(f"{token_id:5} -> {raw_bytes!r} -> {tokenizer.decode([token_id])!r}")

    decoded = tokenizer.decode(ids)
    print(f"Decoded: {decoded!r}")
    print(f"Exact round trip: {decoded == text}")
    # A special spelling must be explicitly allowed to act as a special token.
    end_ids = tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})
    print(f"End-of-text IDs: {end_ids}")
    assert ids == expected_ids, "The exercise IDs differ from the expected GPT-2 IDs."
    assert decoded == text, "Decoding must recover the original text."


if __name__ == "__main__":
    main()
