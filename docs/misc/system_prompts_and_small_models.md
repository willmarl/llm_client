# System Prompts and Small Models

Small language models (e.g., Qwen3:4b, SmolLM2) often **completely ignore system prompts** or follow them inconsistently. This is a known limitation of models with fewer parameters — they lack the instruction-following capability that larger models have.

**Why this happens:**

- System prompts require the model to understand and prioritize meta-instructions about how to behave
- Smaller models have less capacity to learn these complex instruction-following patterns during training
- The model may see the system message but weight it lower than user content

**Solutions:**

1. **Use a larger model** — If running locally via Ollama, try `llama2`, `mistral`, `neural-chat`, or other larger models
2. **Use a commercial API** — OpenAI, Anthropic, etc. provide well-tuned models that respect system prompts reliably
3. **Reinforce in user message** — As a workaround, include the system instructions in the first user message (less elegant but can work)

If you're using a small model and the system prompt isn't working, switching to a larger model or different provider is usually the best fix.
