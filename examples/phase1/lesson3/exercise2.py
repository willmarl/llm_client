# Structured Output with Validation
import sys
from pathlib import Path
from pydantic import BaseModel, Field, field_validator, ValidationError

# sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src import get_llm
from rich import print

model = get_llm()


class BookReview(BaseModel):
    title: str = Field(description="The book title")
    rating: int = Field(
        description="Rating from 1 to 5",
        ge=1,
        le=5,  # Must be between 1 and 5
    )
    sentiment: str = Field(
        description="One of: positive, neutral, negative"
    )
    review: str = Field(
        description="A short review (max 100 words)",
        max_length=500,  # Max 500 characters
    )

    @field_validator("sentiment")
    @classmethod
    def validate_sentiment(cls, v):
        allowed = {"positive", "neutral", "negative"}
        if v.lower() not in allowed:
            raise ValueError(f"sentiment must be one of {allowed}")
        return v.lower()


structured_model = model.with_structured_output(BookReview)

try:
    response = structured_model.invoke("Review the book 1984 by George Orwell.")
    print("✓ Valid response:")
    print(response)
except ValidationError as e:
    print("✗ Validation error occurred:")
    print(e)
