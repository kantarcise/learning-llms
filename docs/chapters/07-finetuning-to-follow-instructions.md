# Chapter 7 — Finetuning to Follow Instructions

We built a GPT architecture and [loaded pretrained weights into it](05-pretraining-on-unlabeled-data.md). In [Chapter 6](06-finetuning-for-text-classification.md), each SMS came with one correct label: spam or not spam. In Chapter 7, we start again from the pretrained language model. Each training example gives it an instruction and an example of the response we want it to write.

<figure markdown="1">

[![Raschka’s roadmap branches from a foundation model to a classifier above and a personal assistant below.](https://sebastianraschka.com/images/LLMs-from-scratch-images/mental-model.jpg)](https://sebastianraschka.com/images/LLMs-from-scratch-images/mental-model.jpg)

<figcaption markdown="span">Figure 7.1 places us at Stage 3 of [Sebastian Raschka’s roadmap](https://sebastianraschka.com/llms-from-scratch/). We completed step 8 on the upper branch with labeled SMS messages. Follow the lower branch for step 9, where an instruction dataset teaches the model to respond to requests.</figcaption>
</figure>

For example, suppose the request is “Rewrite this appointment reminder in a friendlier tone.” A training example includes the original reminder and a suitable rewrite. The model learns to generate that response one token at a time. This is one way to teach the behavior people expect from a chatbot or personal assistant.

We’ll prepare the instruction examples, group them into training batches, and fine-tune the pretrained model. Then we’ll give it held-out requests, extract and save the responses it generates, and evaluate how well those responses carry out the instructions.

## 7.1 Introduction to instruction fine-tuning

Notes to come as we read this section.

## 7.2 Preparing a dataset for supervised instruction fine-tuning

Notes to come as we read this section.

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
