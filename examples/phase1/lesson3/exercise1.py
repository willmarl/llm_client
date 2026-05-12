# Structured Output - Movie Recommendation with nested actors
import sys
from pathlib import Path
from typing import List
from pydantic import BaseModel, Field

# sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src import get_llm
from rich import print

model = get_llm()


class Actor(BaseModel):
    name: str = Field(description="The actor's name")
    role: str = Field(description="The character they play")


class MovieRecommendation(BaseModel):
    title: str = Field(description="The movie title")
    genre: str = Field(description="The movie genre")
    rating: float = Field(description="The movie rating (0-10)")
    summary: str = Field(description="A brief summary of the movie")
    actors: List[Actor] = Field(description="List of main actors and their roles")


structured_model = model.with_structured_output(MovieRecommendation)

response = structured_model.invoke("Recommend me a sci-fi movie.")

print(response)

# Without good descriptions:
# class Task(BaseModel):
#     priority: str  # Model might generate: "urgent", "ASAP", "critical", "1"
# The LLM has no constraints—it could pick any priority value.

# With good descriptions:
# priority: str = Field(description="One of: low, medium, high")
# Now the LLM knows the exact values allowed. It will reliably generate "low", "medium", or "high"—not random variations.

# Another example:
# title: str = Field(description="Short task title")
# # Guides it to be concise
# # vs just: title: str
# # # Might generate a paragraph
