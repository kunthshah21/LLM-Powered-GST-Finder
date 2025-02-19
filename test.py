import openai
from dotenv import load_dotenv
import os

load_dotenv()

client = openai.OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Tell me a fun fact about space."}]
)

print(response.choices[0].message.content)
