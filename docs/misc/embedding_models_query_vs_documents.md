## Embedding Models: `embed_query` vs `embed_documents` — The Truth

### What Most Docs Don't Tell You

**It's prompt engineering, not magic.**

LangChain's `embed_query` and `embed_documents` separate methods exist for two reasons:

1. **Batching convenience** — `embed_documents` takes a list (efficient indexing), `embed_query` takes a single string (search)
2. **Instruction-tuned models** — some models (E5, Cohere V3, Nomic) expect different _prefixes_ for queries vs documents

### How Instruction-Tuned Embeddings Actually Work

| Role     | What gets embedded                          |
| :------- | :------------------------------------------ |
| Query    | `"query: What produces energy?"`            |
| Document | `"passage: Mitochondria is the powerhouse"` |

The model was **trained** to map `"query: ..."` vectors close to relevant `"passage: ..."` vectors, even when the raw text differs.

### The Uncomfortable Truth

**This is prompt engineering for embedding models.**

|                                  | LLM Tool Calls                     | Instruction Embeddings |
| :------------------------------- | :--------------------------------- | :--------------------- |
| Mechanism                        | `"You are a helpful assistant..."` | `"query: ..."`         |
| Requires training on pattern     | ✅                                 | ✅                     |
| Fails on small/dumb models       | ✅                                 | ✅                     |
| Without training, prefix = noise | ✅                                 | ✅                     |

No one advertises this because "prompt engineering" sounds less impressive than "instruction-tuned asymmetric embedding models."

### Bottom Line

- **Basic models** (OpenAI `text-embedding-3-small`, `all-MiniLM-L6-v2`): `"query: "` prefix does nothing useful — the model wasn't trained on it. Implement `embed_query` as `embed_documents([text])[0]`.
- **Instruction-tuned models** (E5, Cohere V3, Nomic, Qwen3): Prefix matters — but the model must be large/smart enough to have learned the pattern during training.

### Key Insight

Adding `"query: "` isn't organizing files with `"0_"` prefixes (alphabetical sorting). It's a **training signal** that activates different embedding pathways — but only if the model was actually trained to recognize it. Small models ignore it completely.
