# LLM

## Initalize LLM

> `get_llm()`

> more sophisticated/intended use case example

```python
from llm_client import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

system_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{text}")
])

answer_parser = StrOutputParser()

llm = get_llm()
chain = RunnableParallel(
    answer = system_prompt | llm | answer_parser
)

query = "Why is the sky blue"

result = chain.invoke({"text": query})
print(result)
```

> could technically do this ⬇️ but just use `run_llm()` instead

```python
from llm_client import get_llm

response = get_llm().invoke("Hello, how are you?")

print(response)
```

**Description**:
Initialize LLM for langchain. Gets the LLM based on provider and model from config.

**Parameters**:

- None

**Returns**:

- LLM instance (ChatOpenAI, ChatAnthropic, or ChatOllama)

> returned `response` from 2nd example
>
> - `<class 'langchain_openai.chat_models.base.ChatOpenAI'>`
> - _quick visual hierachy for **openai**, will differ with other models_

```
├─ content (str)
├─ additional_kwargs (dict)
│  └─ refusal
├─ response_metadata (dict)
│  ├─ token_usage (dict)
│  │  ├─ completion_tokens, prompt_tokens, total_tokens
│  │  ├─ completion_tokens_details (dict)
│  │  └─ prompt_tokens_details (dict)
│  └─ model_provider, model_name, system_fingerprint, id, service_tier, finish_reason, logprobs
├─ id (str)
└─ usage_metadata (dict)
   ├─ input_tokens, output_tokens, total_tokens
   ├─ input_token_details (dict)
   └─ output_token_details (dict)
```

## Get LLM response

> `run_llm(prompt: str)`

```python
from llm_client import run_llm

response = run_llm("What is the capital of France?")

print(f"Response: {response}")
```

**Description**:
Run the LLM with the given prompt and return the response text.

**Parameters**:

- prompt (str): The input prompt for the LLM.

**Returns**:

- str: The response from the LLM.
