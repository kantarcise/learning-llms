# Chapter number — title

Source and sections/pages:

## Opening context

What are we about to understand, and why does it matter?

## Section title from the book

Explain the idea in your own words. Use as many subsections as the concept needs, rather than repeating a fixed form for every section.

### Let's unpack it

Define unfamiliar terms and connect them to something already understood.

### A small example

Show inputs, shapes, steps, and expected results. Link the runnable version when available.

### A useful tangent

Optional: investigate a question that arose while reading, then connect back to the main topic.

## Exercises and experiments

Record attempts, explanations, run commands, and observations.

## End of chapter

How do the ideas fit together? What can I explain now? What remains unclear?

## References

## Figure convention

Use Material's [image-caption support](https://squidfunk.github.io/mkdocs-material/reference/images/#image-captions) for every explanatory image. Keep descriptive alt text, place the explanation directly below the image, and attribute external figures. Link large diagrams to their full-size image when useful. The configured `md_in_html` extension supports this pattern without an extra dependency:

```html
<figure markdown="1">

![Describe what the diagram shows](path-to-image.svg)

<figcaption markdown="span">Explain the figure and link its source when applicable.</figcaption>
</figure>
```

## Code-block titles

Use code-block titles sparingly, for substantial class definitions or module-sized examples where naming the component helps the reader. Leave one-liners, short explanatory snippets, and step-by-step calculations untitled; introduce their source in the surrounding prose when needed. Add `title="..."` after the language on the code fence for examples that benefit from a title. Use the real repository path and function name for source excerpts. For illustrative snippets that do not live in a module, use a descriptive title rather than inventing a filename. See [Material's code-block title documentation](https://squidfunk.github.io/mkdocs-material/reference/code-blocks/#adding-a-title).
