# Chapter 1 — Understanding Large Language Models

Status: reading in progress, October 16, 2026. Notes cover the opening discussion in 1.1 and a brief observation from 1.2. I am now starting 1.3, “Stages of building and using LLMs.” The chapter is not complete.

The numbered headings follow the book so I can move between the reading and this companion. Smaller headings within each section organize our explanations and tangents; they are not additional book subsections.

These notes paraphrase ideas from Chapter 1 of Sebastian Raschka's *Build a Large Language Model (From Scratch)*, with our discussion and preparation experiments alongside them. They are not a transcript or a reproduction of the book's figure. See the [author's official companion repository](https://github.com/rasbt/LLMs-from-scratch).

## Before the chapter: the Python foundation 📘

In “About this book,” the emphasis on a strong Python foundation made me feel good about the time I invested in *Fluent Python* and *Data Structures and Algorithms in Python*. I have already taken steps toward that prerequisite. The new challenge is connecting familiar programming ideas to tensors, model architecture, and training.

## 1.1 What is an LLM?

### What do we mean by “understanding”?

One of the first distinctions that stood out to me is how the book uses the word **understand**. In this context, it describes a model processing and generating language in coherent, relevant ways. That is a description of behavior; it is not, by itself, evidence of human-like consciousness or comprehension.

Consciousness is a tricky subject, and I am not sure how this discussion will develop in the future. For these notes, I want to keep the practical meaning separate from the philosophical question. We can examine what a model does, how it is trained, and where it fails without treating fluent language as an answer to whether it has subjective experience.

This distinction is useful after our preparation experiments. Our Cosmopedia model started producing recognizable phrases, but also repeated them. We observed learned language patterns; that observation alone does not settle what kind of “understanding” the model has.

### From specialized NLP tasks to broader language capabilities

The contrast with earlier natural language processing (NLP) applications stood out to me. Many systems were designed and trained for particular tasks: sentiment analysis, translation, classification, and so on. LLMs can support a broader range of language tasks through the same underlying model, often guided by a prompt or additional training.

This is a shift in breadth, not a claim that all earlier NLP was limited to one task or that an LLM performs every task reliably. NLP is still the field concerned with computational work on human language; LLMs are one family of approaches within it.

### Where do these fields fit? 🧩

The basic nesting is **artificial intelligence → machine learning → deep learning**. AI is the broad field of systems performing tasks associated with intelligence; it does not require a claim of human-like consciousness. Machine learning learns patterns from data, although people still write the training code and choose the objectives. Deep learning uses neural networks with multiple layers of learned computation.

![AI contains machine learning, which contains deep learning. Modern generative deep-learning models include generative LLMs and image or audio generators; CNN classifiers are another deep-learning example.](../assets/chapter-01/ai-map.svg)

*Original companion diagram based on the relationships discussed in the opening reading. It focuses on modern deep-learning examples rather than reproducing the book's figure.*

**Generative AI describes producing content; it is not another name for LLMs.** A GPT-like language model produces text, while other generative models produce images or audio. The diagram places the generative LLMs we study inside that broader group. It also leaves room for non-generative deep learning, such as the CNN classifiers I know from earlier work.

The picture is intentionally scoped: generative methods also exist outside deep learning, and language models can be used for classification or representations rather than text generation. The labels describe related fields, model families, and uses; they are not all interchangeable levels of one universal taxonomy.

### Why does next-token prediction teach useful patterns?

The book's explanation connects next-word prediction with the sequential structure of language. To predict a plausible continuation, a model can benefit from learning grammar, topic, context, and relationships expressed in text. In our implementations, the prediction unit is a **token**, which may be a character, a word, or part of a word.

For example, consider this illustrative prefix:

```text
She put the ice cream in the ...
```

A plausible continuation depends on more than the immediately previous word. Earlier words constrain what makes sense. Training repeatedly rewards assigning probability to the actual continuation, which gives the model a reason to learn useful contextual patterns. The objective encourages those patterns; it does not guarantee factual correctness or human-like comprehension.

I see a connection here: we express ourselves through language, and the model learns from patterns in that expression. The sequential prediction task is simple to state, while the relationships needed to perform it well can be much richer.

Our experiments made the training signal concrete. We supplied input tokens and shifted next-token targets. Cross-entropy measured the probability assigned to the correct target; backpropagation and the optimizer updated the learned parameters. See the [character-log comparison](../preparation/training-comparison.md) and [Cosmopedia experiment](../preparation/cosmopedia.md).

### Selective attention in transformers

The attention explanation now connects with calculations we have already followed. At a position, a transformer can form a weighted mixture of information from other accessible positions. Those mixing amounts depend on the representations and learned query/key projections; value vectors supply the information being mixed.

“Selective attention” does not mean a conscious decision. It describes the calculated weighting. In our causal language model, future positions are masked, so a prediction can use its own position and earlier ones. Attention works alongside position information, feedforward networks, normalization, and residual connections.

The book focuses on transformer-based language models. We should read that as the architecture being studied, rather than a definition that every possible language model must use transformers. The original [Attention Is All You Need paper](https://arxiv.org/abs/1706.03762) introduced the Transformer architecture around attention mechanisms.

## 1.2 Applications of LLMs

The application that stood out in this section is chatbots: using an LLM to generate responses in a conversation. This is a brief reading note, not an exhaustive list of the section's applications.

## 1.3 Stages of building and using LLMs

**Current reading position.** I have reached this section; detailed notes will follow as I read and bring observations or questions. No section summary has been filled in ahead of the reading.

## Questions to carry forward

- How should I distinguish useful language competence from the different meanings of “understanding”?
- As I read, how does next-token prediction connect to the broader abilities demonstrated by larger models?

The consciousness question is worth keeping, but we do not need to resolve it to follow the architecture. For now, continue the chapter and bring the next passages that stand out.

## End of chapter

Not reached yet. No chapter exercises or end-of-chapter conclusions are recorded as completed.
