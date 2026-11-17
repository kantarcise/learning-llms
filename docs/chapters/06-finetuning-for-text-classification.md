# Chapter 6 — Finetuning for Text Classification

At the end of [Chapter 5](05-pretraining-on-unlabeled-data.md), we loaded OpenAI’s pretrained GPT-2 weights and asked the model to continue a sentence. Here we keep what GPT-2 learned about language and change the question. Given a complete text message, can it decide whether the message is spam?

<figure markdown="1">

[![Book roadmap showing the pretrained model branching into a classifier and a personal assistant.](https://sebastianraschka.com/images/LLMs-from-scratch-images/mental-model.jpg)](https://sebastianraschka.com/images/LLMs-from-scratch-images/mental-model.jpg)

<figcaption markdown="span">In Figure 6.1 we reach Stage 3 of [Sebastian Raschka’s roadmap](https://sebastianraschka.com/llms-from-scratch/). Follow the upper branch to the classifier in Chapter 6. The instruction-following branch leads to [Chapter 7](07-finetuning-to-follow-instructions.md).</figcaption>
</figure>

We’ll prepare labeled messages, adapt GPT-2 to produce a score for each class, and fine-tune it on those examples. Then we’ll measure accuracy on held-out messages and use the classifier on a new SMS.

## 6.1 Different categories of fine-tuning

Imagine a text arrives saying **“Claim your free cruise by midnight!”** What would one training example look like? Figures 6.2 and 6.3 in the book show two ways to use a pretrained model. Here they are side by side with the same message.

| Fine-tuning approach | What goes into the model | Target we train it to produce |
| --- | --- | --- |
| Instruction | “Is this spam? Answer yes or no.” + the cruise message | The text **Yes** |
| Instruction | “Translate this into Turkish.” + the cruise message | The text **Ücretsiz gemi seyahatinizi gece yarısına kadar talep edin!** |
| Classification | The cruise message alone | The class **spam** |

In **instruction fine-tuning**, each example pairs a prompt with a desired response. The instruction is part of the input. Changing just that instruction can turn a yes-or-no question into a translation task. The model learns by predicting the tokens of the desired response.

In **classification fine-tuning**, each example pairs a message with one label. The task stays the same across messages. “Can you meet me at the ferry?” would still go through the same two outputs, one score for *spam* and one for *not spam*. The higher score supplies the predicted label. If you’ve trained a CNN to identify plant species, this is the familiar part: it can recognize a new specimen, and its answer comes from the species you defined for training.

!!! tip "What makes this a yes-or-no decision?"

    Both systems could put the same message in a spam folder. In the instruction example, “answer yes or no” asks the model to express its decision in words. In the classifier, the two scores are built into the output layer. The application can act on the selected class directly.

!!! note "Choosing the right approach"

    For a spam filter, labeled messages give us exactly the examples we need, and we can check predictions against known labels. For an assistant that must translate, summarize, and answer questions, instruction–response pairs let us teach each of those tasks. Covering that broader range generally takes more varied data and more training compute than specializing the same starting model for a single task.

## 6.2 Preparing the dataset

Figure 6.4 divides the work into three stages. First we prepare labeled examples and group them into batches. Then we set up pretrained GPT-2 with an output for each class. Finally we train, evaluate, and classify new messages. This section takes us from a downloaded archive to three CSV files. The same workflow applies to email if we supply labeled emails.

**What is one example?** Figure 6.5 previews a table with two columns. Each row holds one complete SMS and the label someone assigned to it. *Ham* is the dataset’s term for a legitimate message. Download and unzip the [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection). The archive member `SMSSpamCollection` is already tab-separated text, even though its name has no extension. From the directory containing that file:

```python
import pandas as pd

messages = pd.read_csv("SMSSpamCollection", sep="\t", header=None, names=["Label", "Text"])
print(messages.head())                   # Inspect the pairing before changing anything.
print(messages["Label"].value_counts())  # 4,825 ham and 747 spam.
```

Of the 5,572 messages, nearly 87% are ham. A filter that marked every message as ham would already be about 87% accurate here and miss every spam message. For this small teaching run we keep every spam example and draw an equal number of ham examples at random. This **undersampling** keeps training manageable, at the cost of leaving out 4,078 legitimate messages and changing how common spam looks.

```python
spam = messages[messages["Label"] == "spam"]
ham = messages[messages["Label"] == "ham"].sample(n=len(spam), random_state=123)
balanced = pd.concat([ham, spam], ignore_index=True)
balanced["Label"] = balanced["Label"].map({"ham": 0, "spam": 1})
```

**A class ID names the answer for the whole message.** Mapping `spam` to `1` resembles a vocabulary lookup because a name gets a stable integer. The two lookups serve different jobs. GPT’s tokenizer will turn the **text** into a sequence of input IDs in §6.3. The **label** is already one integer, telling the loss function which of the two class scores should be higher.

```python
# Mix the classes before taking slices. The seed makes this split reproducible.
rows = balanced.sample(frac=1, random_state=123).reset_index(drop=True)
train_end = int(0.70 * len(rows))
validation_end = train_end + int(0.10 * len(rows))

rows.iloc[:train_end].to_csv("train.csv", index=False)
rows.iloc[train_end:validation_end].to_csv("validation.csv", index=False)
rows.iloc[validation_end:].to_csv("test.csv", index=False)
```

| File | Messages | What we use it for |
| --- | ---: | --- |
| `train.csv` | 1,045 | Update the model weights. |
| `validation.csv` | 149 | Compare training choices before settling on a model. |
| `test.csv` | 300 | Evaluate the chosen model on held-out examples. |

The full pool contains 747 examples of each class. Shuffling and slicing gives approximate 70/10/20 splits without guaranteeing a perfect class balance inside each file. Since we balanced the pool *before* splitting, the test accuracy will describe a population with roughly equal numbers of spam and ham messages.

## 6.3 Creating data loaders

In Chapter 2, cutting a sliding window gave us equally long rows before batching. Here one SMS is one example, and its target is one class ID. One message might encode to three tokens and another to five. How do we stack both rows in one tensor without cutting off the end of the longer message?

| Message | Encoded input (illustrative IDs) | Input padded to five positions | Class ID |
| --- | --- | --- | --- |
| A | `11 12 13` | `11 12 13 50256 50256` | `0` (ham) |
| B | `21 22 23 24 25` | `21 22 23 24 25` | `1` (spam) |

The inputs now have shape `[2, 5]` and the labels have shape `[2]`. Truncating both messages to three tokens would save computation but discard the last two tokens of B. Padding preserves them and spends two extra input positions on A. The book reuses GPT-2’s end-of-text ID, `50256`, as the right-padding value. It sets the width from the longest **encoded training message** and reuses that width for validation and test, truncating any longer messages in those splits.

!!! tip "Think of letterboxing an image for object detection"

    Imagine fitting a wide photo onto a square canvas. Resizing changes the pixel grid. Letterboxing preserves the aspect ratio during resizing and fills the leftover space with a border. Adding padding IDs is like adding that border: the original token IDs stay in order, and the extra positions let PyTorch stack examples into a rectangular batch. The label still describes the whole message.

### From text to equally long token rows

We can ask the tokenizer which ID represents the special token. The `allowed_special` argument explicitly allows tiktoken to interpret this spelling as a special token.

```python
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))
# [50256]
```

Once a message is encoded, we append the integer `50256` to its list of IDs. Each appended ID occupies one position and will receive an embedding when it enters the model.

<figure markdown="1">

![Three messages encoded with GPT-2 and padded to five positions. Shorter messages receive orange padding cells.](../assets/chapter-06/sms-padding.svg)

<figcaption markdown="span">Our version of Figure 6.6, using three new messages and verified GPT-2 token IDs. Blue cells hold message tokens, orange cells hold padding. The longest row already fills all five positions.</figcaption>
</figure>

### A dataset supplies one message and its label

The class below adapts Listing 6.4. It encodes messages once in its initializer. PyTorch later asks for individual examples through `__getitem__` and the number of examples through `__len__`. Calling `dataset[0]` invokes `__getitem__(0)`. Calling `len(dataset)` invokes `__len__()`.

This listing uses pandas to read the CSV files, alongside PyTorch and tiktoken.

```python title="SpamDataset — prepare and retrieve one labeled SMS"
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset


class SpamDataset(Dataset):
    """Return (token IDs [T], scalar class ID), both torch.long.

    T is max_length, the shared number of token positions per message.
    Labels are 0 for ham and 1 for spam. Each CSV must contain at least
    one message and have Text and Label columns.
    """

    def __init__(
        self,
        csv_file: str | Path,
        tokenizer: tiktoken.Encoding,
        max_length: int | None = None,
        pad_token_id: int = 50256,
    ) -> None:
        self.data = pd.read_csv(csv_file)
        # Each message starts as a separate list with its own token count.
        self.encoded_texts: list[list[int]] = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]

        if max_length is None:
            # Training chooses its width from the longest encoded SMS.
            self.max_length = self._longest_encoded_length()
        else:
            # Validation/test can reuse the width chosen for training.
            self.max_length = max_length
            self.encoded_texts = [
                token_ids[:max_length] for token_ids in self.encoded_texts
            ]

        # A row already at max_length gets zero extra IDs.
        self.encoded_texts = [
            token_ids + [pad_token_id] * (self.max_length - len(token_ids))
            for token_ids in self.encoded_texts
        ]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one padded message [T] and its scalar label."""
        token_ids = self.encoded_texts[index]
        label = int(self.data.iloc[index]["Label"])
        return (
            torch.tensor(token_ids, dtype=torch.long),
            torch.tensor(label, dtype=torch.long),
        )

    def __len__(self) -> int:
        """Return the number of messages available to the data loader."""
        return len(self.data)

    def _longest_encoded_length(self) -> int:
        """Find the largest token count before padding."""
        return max(len(token_ids) for token_ids in self.encoded_texts)
```

```python
train_dataset = SpamDataset("train.csv", tokenizer)
val_dataset = SpamDataset(
    "validation.csv", tokenizer, max_length=train_dataset.max_length
)
test_dataset = SpamDataset(
    "test.csv", tokenizer, max_length=train_dataset.max_length
)

print(train_dataset.max_length)  # 120 for the split from §6.2
token_ids, label = train_dataset[0]
print(token_ids.shape, label.shape)  # torch.Size([120]) torch.Size([])
```

Here 120 counts tokens after encoding. GPT-2 supports up to 1,024 positions. This dataset class has no knowledge of the model’s context limit, so we must choose a width that fits it. Setting `max_length=None` independently on validation or test chooses each split’s own longest row. Each split can still form rectangular batches, provided its chosen width fits the model.

### Exercise 6.1 — Increasing the context length

The exercise asks us to pad to the model’s maximum supported length and observe predictive performance. We can prepare that comparison now by giving every split the same explicit width.

```python
long_train_dataset = SpamDataset("train.csv", tokenizer, max_length=1024)
long_val_dataset = SpamDataset("validation.csv", tokenizer, max_length=1024)
long_test_dataset = SpamDataset("test.csv", tokenizer, max_length=1024)

short_ids, short_label = train_dataset[0]
long_ids, long_label = long_train_dataset[0]
assert len(long_ids) == 1024
assert torch.equal(short_ids, long_ids[:120])
assert short_label.item() == long_label.item()
assert torch.all(long_ids[120:] == 50256)
```

For our training split, this adds **904 padding positions per message**, with the original message and label preserved. The input width grows by about 8.5 times. An ordinary dense attention score matrix grows with the square of that width, about 72.8 times here. Actual runtime and peak memory also depend on the attention implementation and other parts of the model.

**Performance comparison to run after implementing fine-tuning:** train a fresh copy of the same pretrained model at each width, keeping the split, seed, optimizer, batch size, and epochs fixed. Record training time and validation accuracy. Use validation to select the setting, then evaluate the chosen model on the test set. The data preparation and assertions above have been checked. Classification accuracy for these two settings has yet to be measured.

More padding supplies additional positions for computation. Its effect on classification depends on how the classifier uses those positions, which we will see when adding its output head.

## 6.4 Initializing a model with pretrained weights

Notes to come as we read this section.

## 6.5 Adding a classification head

Notes to come as we read this section.

## 6.6 Calculating the classification loss and accuracy

Notes to come as we read this section.

## 6.7 Fine-tuning the model on supervised data

Notes to come as we read this section.

## 6.8 Using the LLM as a spam classifier

Notes to come as we read this section.

## Questions to bring to the chapter

<!-- Add actual questions during reading. -->

## End of chapter

<!-- Summarize what became clear, link exercises, and record unresolved questions. -->
