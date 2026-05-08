## Pick which image extracting method(s) you want to use

> `ultimate_image_extractor(img_path)`

```python
from rich import print
from llm_client import ultimate_image_extractor

img_path = "image.png"
results = ultimate_image_extractor(img_path)

print(type(results))  # <class 'dict'>
print(results)
# {'OCR': [0.123, 9.876, ...], 'multimodal': [0.123, 9.876, ...], 'CLIP': [0.123, 9.876, ...]}

```

**Description**:
By default puts through OCR, multimodal, and CLIP
it first extracts text with OCR and with extracted text converts to embebbedding.
same thing with multimodal captioning
CLIP doesnt output text so its normal embedding
Can also choose which methods you albeit verbose needing to do

**Parameters**:

- str1 `str`: can input either "OCR", "caption", or "CLIP"
- str2 `str`: can input either "OCR", "caption", or "CLIP"
- str3 `str`: can input either "OCR", "caption", or "CLIP"

**Returns** (dict):
if doing `ultimate_image_extractor(img_path)`

```python
{
    'OCR': {'text_results': 'Pic is ...', 'embed_results': [0.123, 9.876]},
    'caption': {'text_results': 'Pic is ...', 'embed_results': [0.123, 9.876]},
    'CLIP': {'text_results': None, 'embed_results': [0.123, 9.876]}
}
```

if doing `ultimate_image_extractor(img_path, "CLIP", "caption")`

```python
{
    "caption": {"text_results": "Pic is ...", "embed_results": [0.123, 9.876]},
    "CLIP": {"text_results": None, "embed_results": [0.123, 9.876]},
}
```

---

# Which to mode(s) to pick?

give quick tldr explaination here on how to think/decide

## Decision Flowchart

```mermaid
graph TD
    Start[What do you need from your images?] --> Question1{Does the image<br/>contain text<br/>you need to read?}

    Question1 -->|Yes| Question2{Is it simple,<br/>printed text?<br/>Forms, receipts, docs}
    Question1 -->|No| Question3{What type of<br/>search do you need?}

    Question2 -->|Yes| Question4{How many images?}
    Question2 -->|No, complex/<br/>handwritten| UseMultimodal[Use Multimodal<br/>Captioning]

    Question4 -->|1000+| UseOCR[Use OCR<br/>Fast & Free]
    Question4 -->|<1000| Question5{Budget flexible?}

    Question5 -->|Yes| UseMultimodal2[Use Multimodal<br/>Better accuracy]
    Question5 -->|No| UseOCR2[Use OCR<br/>Good enough]

    Question3 -->|Visual similarity<br/>colors, style, mood| UseCLIP[Use CLIP<br/>Fast & Cheap]
    Question3 -->|Semantic understanding<br/>what's happening?| Question6{Budget flexible?}

    Question6 -->|Yes| UseMultimodal3[Use Multimodal<br/>Deep understanding]
    Question6 -->|No| UseCLIP2[Use CLIP<br/>Basic understanding]

    Question1 -->|Both text<br/>AND visual| Question7{Budget?}

    Question7 -->|Limited| HybridCLIPOCR[🔥 CLIP + OCR<br/>Best value!]
    Question7 -->|Flexible| HybridCLIPMulti[💎 CLIP + Multimodal<br/>Premium experience]
    Question7 -->|No limit| HybridAll[🎯 All Three<br/>Maximum coverage]

    UseOCR --> Recommendation1[✅ Extract text<br/>✅ Search keywords<br/>✅ Free & fast]
    UseOCR2 --> Recommendation1

    UseCLIP --> Recommendation2[✅ Visual similarity<br/>✅ Text-to-image<br/>✅ Fast & cheap]
    UseCLIP2 --> Recommendation2

    UseMultimodal --> Recommendation3[✅ Deep understanding<br/>✅ Natural language<br/>⚠️ Slow & expensive]
    UseMultimodal2 --> Recommendation3
    UseMultimodal3 --> Recommendation3

    HybridCLIPOCR --> Recommendation4[✅ Visual + Text search<br/>✅ Fast & cheap<br/>✅ Most versatile<br/>⭐ Recommended!]

    HybridCLIPMulti --> Recommendation5[✅ Visual + Semantic<br/>✅ Best search quality<br/>⚠️ Expensive setup]

    HybridAll --> |Recommendation6| Premium[✅ Everything<br/>⚠️ Most expensive<br/>⚠️ Most complex]

    style Start fill:#e1f5ff
    style HybridCLIPOCR fill:#90EE90
    style Recommendation4 fill:#90EE90
    style UseCLIP fill:#ffd700
    style UseOCR fill:#ffd700
    style UseMultimodal fill:#ff6b6b
    style UseMultimodal2 fill:#ff6b6b
    style UseMultimodal3 fill:#ff6b6b
    style Premium fill:#e37dff
```
