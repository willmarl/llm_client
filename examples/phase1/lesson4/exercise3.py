# Custom Output Parser
import sys
from pathlib import Path

# sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from llm_client import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import BaseOutputParser
from rich import print

model = get_llm()


# Build your own transformations
class BulletPointParser(BaseOutputParser):
    """Extracts and formats bullet points from text"""

    def parse(self, text: str):
        # Split by newlines and filter out empty lines
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        # Add bullet point to each line
        bullet_points = [f"• {line}" for line in lines]
        return "\n".join(bullet_points)


prompt = ChatPromptTemplate.from_template("""
List 3 key features of {topic}.
Keep each feature on a new line.
""")

parser = BulletPointParser()

messages = prompt.invoke({"topic": "Python programming"})

response = model.invoke(messages)

print("Raw response:")
print(response.content)
print("=" * 20)

parsed = parser.invoke(str(response.content))

print("Parsed (with bullets):")
print(parsed)
