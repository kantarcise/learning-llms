# Chapter 2 exercises

## Exercise 2.1 — inspect GPT-2 token pieces

Run from the repository root:

```bash
uv run --locked --group book python practice/chapter-02/exercise_2_1.py
```

The script prints IDs and byte pieces for `Akwirw ier`, checks the supplied
expected IDs, checks exact decoding, and shows GPT-2's end-of-text ID.
The first run may download tokenizer data into tiktoken's cache. No API key
or model weights are needed.

Source: section 2.5 and Exercise 2.1 of Sebastian Raschka's *Build a Large
Language Model (From Scratch)*; see the [official Chapter 2 code](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch02).
Our script adds explanations, version reporting, and explicit checks. The
companion page records the assistant's run separately from user practice.
