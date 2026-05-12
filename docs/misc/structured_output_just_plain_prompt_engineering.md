# Structured Output: Just plain prompt engineering

### 1: Tool calls aren't real

**LLMs only output text.** "Tool calls" are just formatted text (JSON/XML/special tokens) that the API _pretends_ is a function call. No magic - just pattern matching.

### 2: One request, not multiple

LangChain makes **1 API call** → receives 1 response → parses it. No back-and-forth, no multiple messages.

### 3: Failure isn't network errors

You thought tool calls fail due to internet outages or bit flips. **Wrong.** They fail because the model outputs the **wrong text pattern** (or ignores format instructions entirely).

### 4: Small models can work if trained right

A 1.3B **code-trained** model can outperform a 7B general model at JSON output. It's not about size - it's about **whether the training data contained your format** (brackets, braces, tags).

### 5: Same problem as system prompts

Small models ignore tool calls for the **exact same reason** they ignore system prompts - they never learned that those tokens have special meaning.

### 6: Parsers all the way down

LangChain's parser sits on top of OpenAI's parser, which sits on top of the model's pattern matching. **You're always hoping** the layer below does its job.

### Bottom line

Structured output = prompt engineering + training data + hope. No magic.

Here's a concise summary for your notes:
