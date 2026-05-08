**LLM settings**
LLM provider are text so any of trio can do it.

**Text embedding**
Only openai and ollama has text embedders

**Image model**
Image is misleading its actually multimodal.

**Image embedding**
only supports CLIP. Getting it to work locally is tricky to use GPU.
Recommend using replicate if you want something to work now but ofc the trade off is its not local. bye bye privacy.

For faster/cheap use lower res CLIP https://replicate.com/lucataco/clip-vit-base-patch32
For better use offical CLIP https://replicate.com/openai/clip
