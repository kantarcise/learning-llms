# Step Zero

Goal: establish a baseline and get comfortable reasoning about tensors before the book arrives. Budget: six hours. No GPU is needed.

## 1. Baseline — one hour

Write from memory. Uncertainty is useful evidence; leave gaps visible.

- What happens between a text prompt and a generated token?
- What is a tensor, and what does its shape tell me?
- What do I think attention does?
- What might consume memory during generation?
- Which queues, retries, and backpressure mechanisms do I already understand from data-platform work?

My answers:

<!-- Write your own baseline here before looking up explanations. -->

## 2. Tensor practice — two hours

Practise creating tensors, indexing, reshaping, broadcasting, and matrix multiplication on CPU. Record the environment and run command with your exercise.

Before executing each example, predict its shape and values:

- A has shape `[2, 3]`; B has shape `[3, 4]`. What does `A @ B` produce?
- Why does `B @ A` fail for these shapes?
- How does adding a vector of shape `[4]` to a matrix of shape `[2, 4]` work?
- How does reshaping differ from transposing?

Link to my code and explanations:

<!-- Add a relative link after creating your exercise. -->

## 3. Generation loop — one hour

Inspect a generation example from the [book's companion resources](https://sebastianraschka.com/llms-from-scratch/). Run it if practical on your machine; otherwise trace it on paper.

Explain: input tokens → model → next-token scores → selected token → repeat. List what you cannot yet explain.

## 4. Daily capacity — one hour

Given initial cluster capacities and daily usage records, return remaining capacity per day. Each day resets to the original capacities.

Clarify missing records, output ordering, duplicate records, and excessive usage before implementation. Start with a tiny hand-worked example. This is an original practice exercise inspired by the resource-accounting problems discussed during preparation.

## 5. Explain and reflect — one hour

Explain queues, retries, and backpressure in a system I know. What might change for GPU workloads?

- What I understand:
- What I am unsure about:
- What I tested:
- What changed my understanding:

## Completion check

- [ ] Baseline recorded without assistance.
- [ ] Tensor predictions compared with results.
- [ ] Generation loop explained in my own words.
- [ ] Daily-capacity exercise attempted and reviewed.
- [ ] Remaining questions recorded.
