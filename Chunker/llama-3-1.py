# Example: reuse your existing OpenAI setup
from openai import OpenAI

# Point to the local server
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

RULES = "Always answer on russian"
QUESTION = "Introduce yourself."

completion = client.chat.completions.create(
  model="model-identifier",
  messages=[
    {"role": "system", "content": RULES},
    {"role": "user", "content": QUESTION}
  ],
  temperature=0.7,
)

print(completion.choices[0].message.content)
