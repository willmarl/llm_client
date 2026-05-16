# Using llm_client in a new project

## 1. System dependencies (one-time, if not already done)

```bash
sudo apt-get install tesseract-ocr
```

Ollama should already be running if you're using local models.

## 2. Create your project

```bash
mkdir my_project && cd my_project
uv python pin 3.12
uv venv
source .venv/bin/activate
```

## 3. Install torch first (before everything else)

**AMD (ROCm):**
```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2
```

**NVIDIA:**
```bash
uv pip install torch torchvision
```

## 4. Install llm_client and CLIP

```bash
uv pip install /home/cat/repos/llm_client
uv pip install git+https://github.com/openai/CLIP.git
```

> Use `-e` flag if you want changes to llm_client to apply immediately:
> `uv pip install -e /home/cat/repos/llm_client`

## 5. Set up your .env

Copy the template and fill in your keys:

```bash
cp /home/cat/repos/llm_client/.env.template .env
```

Edit `.env` — at minimum set your provider and API key:

```
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-proj-xxx
```

## 6. Use it

```python
from llm_client import get_llm, run_llm, Ingest, load_any_file

response = run_llm("Hello!")
```

The library reads config from the `.env` file in whatever directory you run your script from.
