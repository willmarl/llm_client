from typing import Literal
from ..llm.image_spawner import (
    generate_image_caption,
    generate_image_ocr_text,
)
from .image_embedding import generate_image_embeddings
from .text_embedding import generate_text_embeddings

myArgs = Literal["OCR", "caption", "CLIP", "ocr", "clip"]


def ultimate_image_extractor(
    img_path: str,
    str1: myArgs | None = None,
    str2: myArgs | None = None,
    str3: myArgs | None = None,
):
    """
    Returns dict of results with generator names as keys
    Defaults to running all generators if not specified
    """

    gen_funcs = {
        "OCR": False,
        "caption": False,
        "CLIP": False,
    }

    args = [str1, str2, str3]
    noneCount = 0
    for index, i in enumerate(args):
        if i != None:
            args[index] = i.lower()
        elif i == None:
            noneCount += 1

    # attempt 1 of modular args to do stuff like ultimate_image_exractpr("foo.png", "clip", "caption") or ("foo.png", "CLIP") or ("foo.png")
    if noneCount == 3:
        gen_funcs = {
            "OCR": True,
            "caption": True,
            "CLIP": True,
        }
    else:
        if "ocr" in args:
            gen_funcs["OCR"] = True
        if "caption" in args:
            gen_funcs["caption"] = True
        if "clip" in args:
            gen_funcs["CLIP"] = True

    total_results = {}
    if gen_funcs["OCR"] == True:
        # my_text_results = generate_image_ocr_text(img_path)
        my_text_results = "Pic is ..."
        # true_results = generate_text_embeddings([my_text_results])
        my_embed_results = [0.123, 9.876]

        total_results["OCR"] = {
            "text_results": my_text_results,
            "embed_results": my_embed_results,
        }

    if gen_funcs["caption"] == True:
        # my_text_results = generate_image_caption(img_path)
        my_text_results = "Pic is ..."
        # my_embed_results = generate_text_embeddings([my_text_results])
        my_embed_results = [0.123, 9.876]

        total_results["caption"] = {
            "text_results": my_text_results,
            "embed_results": my_embed_results,
        }

    if gen_funcs["CLIP"] == True:
        # results = generate_image_embeddings(img_path)
        my_embed_results = [0.123, 9.876]
        total_results["CLIP"] = {
            "text_results": None,
            "embed_results": my_embed_results,
        }

    return total_results


# # Usage
# ultimate_image_extractor("/foo.png")  # runs all three, returns array
# ultimate_image_extractor("/foo.png", {'OCR': True, 'multimodal': False, 'CLIP': True})  # selective


# ############
# from typing import TypedDict
# from ..llm.image_spawner import (
#     generate_image_caption,
#     generate_image_ocr_text,
# )
# from .image_embedding import generate_image_embeddings
# from .text_embedding import generate_text_embeddings


# class Generators(TypedDict, total=False):
#     OCR: bool
#     multimodal: bool
#     CLIP: bool


# # def ultimate_image_extractor(
# #     img_path, generators: Generators | None = None
# # ) -> dict[str, list]:
# def ultimate_image_extractor(img_path, generators: Generators | None = None):
#     """
#     generators: dict like {'OCR': True, 'multimodal': False, 'CLIP': True}
#     Returns dict of results with generator names as keys
#     Defaults to running all generators if not specified
#     """

#     gen_funcs = {
#         "OCR": True,
#         "multimodal": True,
#         "CLIP": True,
#     }

#     if generators != None:
#         for keys in generators:
#             gen_funcs[keys] = generators[keys]

#     total_results = {}
#     if gen_funcs["OCR"] == True:
#         text_results = generate_image_ocr_text(img_path)
#         true_results = generate_text_embeddings([text_results])

#         total_results["OCR"] = true_results

#     if gen_funcs["multimodal"] == True:
#         text_results = generate_image_caption(img_path)
#         true_results = generate_text_embeddings([str(text_results)])

#         total_results["multimodal"] = true_results

#     if gen_funcs["CLIP"] == True:
#         results = generate_image_embeddings(img_path)
#         total_results["CLIP"] = results

#     return total_results


# # # Usage
# # ultimate_image_extractor("/foo.png")  # runs all three, returns array
# # ultimate_image_extractor("/foo.png", {'OCR': True, 'multimodal': False, 'CLIP': True})  # selective
