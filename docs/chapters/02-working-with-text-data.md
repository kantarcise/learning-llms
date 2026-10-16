# Chapter 2 — Working with Text Data

Status: reading in progress. Opening notes cover the chapter overview and 2.1, “Understanding word embeddings.” Section 2.2 and later sections are still ahead; no exercises are recorded as completed.

These notes follow Sebastian Raschka's *Build a Large Language Model (From Scratch)*, with explanations from our discussion and links to our experiments. See the [author's Chapter 2 code](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch02).

The chapter introduces text preparation, word and subword tokenization, byte pair encoding (BPE), sliding-window training examples, and the vectors supplied to a language model. These are the steps before the transformer blocks we explored during preparation.

## 2.1 Understanding word embeddings

Text stored in a computer is already digital. The additional problem is giving a neural network a numerical representation on which useful computations can be learned. Characters and token IDs identify categories; their arbitrary numeric IDs do not measure meaning.

My earlier analogy was “digitizing something.” That captured the bridge from data to numbers, but it needs refining: **an embedding maps an input to a vector representation**, usually learned in the neural-network settings we are discussing. It is not analog-to-digital conversion. Audio samples and image pixels can already be numeric before an embedding model processes them.

### Token IDs are addresses; embeddings are vectors

Our character model makes the distinction concrete:

```text
Text character    Token ID       Looked-up vector
     j      →        12     →    [0.12, -0.40, 0.08, 0.91, ...]
                                  32 coordinates in total
```

The vector values above are illustrative, not a checkpoint dump. ID 12 selects row 12 of the learned embedding table. It does not mean that `j` has a magnitude of twelve, or that the next ID represents a similar character.

The table begins randomly initialized and is updated through training. Looking up a row during a forward pass does not itself update it. Subsequent transformer blocks produce contextual representations from these starting vectors.

**Vocabulary size tells us how many rows. Width tells us how many coordinates per row.** Our [character-log model](../preparation/training-comparison.md) has 25 rows and 32 coordinates per row. The [Cosmopedia model](../preparation/cosmopedia.md) has 49,152 rows and 256 coordinates per row.

### Does every model need a separate embedding model?

No separate service or pretrained embedding model was needed for either of our language models. Their `nn.Embedding` layers are part of the models themselves and are trained with the rest of the parameters. Reusing a tokenizer in Cosmopedia did not import pretrained embedding weights.

An external text-embedding model serves a different use case: it can turn a whole query or passage into a vector for comparison and retrieval. An image or audio encoder can likewise produce learned representations from its own input format. Some multimodal systems train representations into a shared space, so different data types do not necessarily require unrelated, separately deployed models.

Not every machine-learning system uses learned dense embeddings. A system might use hand-designed numeric features or sparse word counts. The key distinction here is between an embedding layer inside a language model and a model whose output is an embedding for another application.

### Words, sentences, passages, and documents

The represented unit matters:

| Representation | What one vector represents | Example use |
| --- | --- | --- |
| Token embedding | One vocabulary entry, such as a character or subword | Starting representation inside our GPT-like models |
| Contextual token representation | One position after processing surrounding accessible context | Next-token prediction |
| Sentence or passage embedding | A span of text | Comparing a query with candidate passages |
| Document embedding | A document, within the embedding model's input limits | Document similarity or retrieval |

A sentence embedding is not simply the token ID of a longer string. An encoder processes the text and produces a representation, for example through pooling. Long documents may need splitting into passages rather than squeezing everything into one vector. Similarity is useful only to the extent that the representation and its training objective support the comparison.

### A brief RAG connection

I wondered whether retrieval-augmented generation had faded along with the hype. There is concrete evidence of continued use: [Claude's project documentation](https://support.claude.com/en/articles/11473015-retrieval-augmented-generation-rag-for-projects) describes RAG for project knowledge, and [Anthropic's context-engineering article](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) discusses retrieval as part of selecting useful model context. These examples establish ongoing use, not an industry-wide adoption trend.

```text
Question → retrieve relevant passages → give passages + question to the LLM
                                     → generate an answer using that context
```

Embedding similarity is one retrieval method. Keyword search and hybrid approaches can also supply passages. Retrieving text does not update the language model's weights. A larger context window may let us include more material directly, but does not automatically eliminate the value of selecting relevant material from a large or changing collection.

For this chapter, the useful distinction is **token vectors inside the LLM versus passage vectors used to find context for it**. Building a RAG system can wait.

### Word2Vec and embedding width

[Word2Vec](https://arxiv.org/abs/1301.3781) introduced efficient ways to learn word vectors from relationships between words and their surrounding context. Its common formulations include predicting a word from its context and predicting context words from a word. A conventional Word2Vec table gives a word a fixed vector; a transformer can produce different contextual representations for that word in different sentences.

Yes, the dimensionality discussed here connects directly to our earlier questions about vector width. More coordinates can give a representation more capacity, but do not guarantee better results. Training data, objective, model design, and compute budget matter too. Individual coordinates are not normally assigned human-readable meanings.

For a vocabulary of 25 entries:

| Chosen width | Token embedding table | Stored numbers |
| --- | --- | ---: |
| 32 | 25 × 32 | 800 |
| 64 | 25 × 64 | 1,600 |

Changing width has consequences beyond the embedding table. A square attention projection grows from 32 × 32 = 1,024 weights to 64 × 64 = 4,096 weights when width doubles. This is why width can make later matrix computations grow much faster than the embedding-table size. Vocabulary size and width remain independently chosen dimensions.

## 2.2 Tokenizing text

This is the next section to develop as reading continues. The chapter separates the conceptual introduction to embeddings from the practical process of splitting text into tokens. No tokenization exercises are recorded as completed yet.

## Connection to our experiment tooling: checkpoints

The planned [resumable Cosmopedia training work](https://github.com/kantarcise/learning-llms/issues/14) saves and restores model parameters, optimizer state, progress, and randomness/sampling state. It does not require changing layers or the forward computation, so the architecture drawings remain valid. Weight values may change with training while shapes stay the same.

If we later change the number of blocks, vector width, vocabulary, or another architectural choice, we should update the drawings separately. Current inference-only checkpoints cannot restore optimizer state that was never saved.

## Questions to carry forward

- How does BPE choose useful subword pieces?
- How do sliding windows produce input/target pairs from a longer text?
- How will the book's embedding implementation connect to our existing `nn.Embedding` lookup?

## End of chapter

Not reached yet. Continue reading; this page records the opening discussion, not a completed chapter.
