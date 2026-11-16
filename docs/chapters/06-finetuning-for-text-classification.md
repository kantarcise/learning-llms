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

Notes to come as we read this section.

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
