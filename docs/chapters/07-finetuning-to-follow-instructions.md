# Chapter 7 — Finetuning to Follow Instructions

We built a GPT architecture and [loaded pretrained weights into it](05-pretraining-on-unlabeled-data.md). In [Chapter 6](06-finetuning-for-text-classification.md), each SMS came with one correct label: spam or not spam. In Chapter 7, we start again from the pretrained language model. Each training example gives it an instruction and an example of the response we want it to write.

<figure markdown="1">

[![Raschka’s roadmap branches from a foundation model to a classifier above and a personal assistant below.](https://sebastianraschka.com/images/LLMs-from-scratch-images/mental-model.jpg)](https://sebastianraschka.com/images/LLMs-from-scratch-images/mental-model.jpg)

<figcaption markdown="span">Figure 7.1 places us at Stage 3 of [Sebastian Raschka’s roadmap](https://sebastianraschka.com/llms-from-scratch/). We completed step 8 on the upper branch with labeled SMS messages. Follow the lower branch for step 9, where an instruction dataset teaches the model to respond to requests.</figcaption>
</figure>

For example, suppose the request is “Rewrite this appointment reminder in a friendlier tone.” A training example includes the original reminder and a suitable rewrite. The model learns to generate that response one token at a time. This is one way to teach the behavior people expect from a chatbot or personal assistant.

We’ll prepare the instruction examples, group them into training batches, and fine-tune the pretrained model. Then we’ll give it held-out requests, extract and save the responses it generates, and evaluate how well those responses carry out the instructions.

## 7.1 Introduction to instruction fine-tuning

A pretrained GPT learns to predict the **next token**. If the prompt begins “The alert was sent by Mina,” a plausible continuation might describe what happened after the alert. Now put a request in front of that same sentence: “Rewrite in active voice.” A useful answer has to preserve the event and change its wording: **“Mina sent the alert.”** A fluent continuation can still miss the requested edit.

**Supervised instruction fine-tuning** gives the pretrained model worked examples of that distinction. During training, each instruction comes with a desired response. The model still predicts one token at a time; the examples teach it what kind of text should follow a request. When we later give it a new instruction, it has to generate the response itself.

<figure markdown="1">

![Three instruction–response pairs illustrate a unit conversion, a synonym, and an edit into active voice.](../assets/chapter-07/instruction-pairs.svg)

<figcaption markdown="span">Companion to Raschka’s Figure 7.2. Read each row from left to right. The response demonstrates what counts as following that particular instruction. For the synonym task, several different answers could be valid.</figcaption>
</figure>

<figure markdown="1">

![Three stages with nine ordered steps. The first step, downloading and formatting instruction examples, is highlighted.](../assets/chapter-07/instruction-tuning-stages.svg)

<figcaption markdown="span">Companion to Raschka’s Figure 7.3. Follow each stage from left to right, then move down a row. Inspecting loss shows how training is progressing. Reading and scoring generated answers tells us how well the model follows the requests.</figcaption>
</figure>

## 7.2 Preparing a dataset for supervised instruction fine-tuning

Raschka prepared 1,100 examples for the book. Download the JSON file and inspect two records before thinking about tokens. Python reads its outer array as a list and each record as a dictionary. [Book dataset](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch07/01_main-chapter-code/instruction-data.json)

```python
import json
from urllib.request import urlopen

url = (
    "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/"
    "ch07/01_main-chapter-code/instruction-data.json"
)
with urlopen(url, timeout=30) as response:
    data = json.load(response)

print(len(data))  # 1100
print(data[50])   # Spelling task with a separate word to correct
print(data[999])  # Antonym question with an empty input field
```

In the spelling record, the **instruction** asks for a correction, the **input** supplies `Ocassion`, and the **output** gives the corrected spelling, `Occasion`, in a sentence. In the antonym record, the instruction already contains the word *complicated*. An empty `input` makes sense here. The example still has a request and a target answer.

A **prompt template** tells us how to arrange those fields into the sequence the tokenizer will encode. During fine-tuning we include the worked answer. When asking the trained model a new question, we supply the request up to the response marker and let the model generate what follows.

<figure markdown="1">

![One record asks the model to double 12. Alpaca places 12 in an input section. Phi-3 puts 12 in the user turn. Both put 24 in the answer.](../assets/chapter-07/prompt-styles.svg)

<figcaption markdown="span">Companion to Raschka’s Figure 7.4. Follow the number 12 through both formats, then find where the answer 24 begins. The Alpaca excerpt omits its introductory sentence for space. [Stanford Alpaca templates](https://github.com/tatsu-lab/stanford_alpaca#data-release) · [Microsoft Phi-3 prompt format](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct#input-formats)</figcaption>
</figure>

!!! tip "What changes with another model?"
    Keep the same request and answer. **Gemma 2 IT** opens the answer turn with `<start_of_turn>model` and closes it with `<end_of_turn>`. **Qwen2.5-7B-Instruct** opens it with `<|im_start|>assistant` and closes it with `<|im_end|>`. In both examples the answer text, **24**, follows the opening marker on a new line. These are excerpts. For an existing instruction-tuned model, use its tokenizer’s chat template to insert the full sequence of markers and line breaks it learned during training. [Gemma prompt formats](https://ai.google.dev/gemma/docs/core/prompt-structure) · [Qwen tokenizer and usage example](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/tokenizer_config.json)

The chapter uses Alpaca as its baseline. Stanford published its training recipe and templates, so we can see exactly how a record becomes training text. In this version of Listing 7.2, `format_input` builds the request. The caller adds the response heading and, for a training example, the known answer. [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) · [Chapter 7 notebook](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch07/01_main-chapter-code/ch07.ipynb)

```python
def format_input(entry: dict[str, str]) -> str:
    sections = [
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.",
        f"### Instruction:\n{entry['instruction']}",
    ]
    if entry["input"]:
        sections.append(f"### Input:\n{entry['input']}")
    return "\n\n".join(sections)


entry = data[50]
prompt = format_input(entry) + "\n\n### Response:\n"
print(prompt + entry["output"])
```

```text
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Identify the correct spelling of the following word.

### Input:
Ocassion

### Response:
The correct spelling is 'Occasion.'
```

Try `data[999]` in the same listing. The antonym question stands on its own, so the response heading follows the instruction directly.

### Exercise 7.1 · Changing the prompt style

Suppose the trained model spells *Occasion* correctly after seeing `### Response:`. Will it still do so after seeing `<|assistant|>`? Start with the **same Alpaca-tuned checkpoint** and the same held-out spelling questions. Ask each question twice, once in each format, with greedy decoding and the same output budget. Count correct spellings and record cases where the model continues the prompt or prints a role marker. This measures how sensitive that checkpoint is to the format used at inference.

For a third version, borrow the single-turn format from Mistral-7B-Instruct v0.1. Its closing `[/INST]` marks where the answer begins. Here are the two alternative prompts to try alongside `prompt` above. The GPT tokenizer will encode these markers as text. [Mistral’s template](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1)

```python
request = "\n".join(part for part in (entry["instruction"], entry["input"]) if part)
phi3_prompt = f"<|user|>\n{request}<|end|>\n<|assistant|>\n"
mistral_prompt = f"[INST] {request} [/INST]"
```

A second experiment asks whether the **training format** matters. Start each run from the same pretrained GPT weights, format the same training examples in one style per run, and evaluate each model in its own style. Keep the split, optimizer settings, seed, and decoding settings fixed. Score the same held-out requests for correctness and instruction following. The formatting examples can run now. The response-quality comparison comes after the fine-tuning run in §7.6.

## 7.3 Organizing data into training batches

Notes to come as we read this section.

## 7.4 Creating data loaders for an instruction dataset

Notes to come as we read this section.

## 7.5 Loading a pretrained LLM

Notes to come as we read this section.

## 7.6 Fine-tuning the LLM on instruction data

Notes to come as we read this section.

## 7.7 Extracting and saving responses

Notes to come as we read this section.

## 7.8 Evaluating the fine-tuned LLM

Notes to come as we read this section.

## 7.9 Conclusions

Notes to come as we read this section.

## Questions to bring to the chapter

<!-- Add actual questions during reading. -->

## End of chapter

<!-- Summarize what became clear, link exercises, and record unresolved questions. -->
