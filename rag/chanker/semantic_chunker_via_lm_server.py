# Example: reuse your existing OpenAI setup
from openai import OpenAI

# Point to the local server
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

RULES = ("Always answer on russian. For each text, that i give you you should respond me with the text, that has only"
         " russian letters, comma and dot sign. You should semantic split this text on parts and return them, "
         "you should not respond with any information that is not included in that text, you should avoid all"
         " unnecessary words and constructions, you should focus only on your task, you will be tipped with 100$"
         " if you make everything like i said")

def ask_question(question):
    completion = client.chat.completions.create(
        model="model-identifier",
        messages=[
            {"role": "system", "content": RULES},
            {"role": "user", "content": question}
        ],
        temperature=0.7,
    )

    return completion.choices[0].message.content


f = open("../data/Henry.txt", "r", encoding='utf-8')
s = f.read()
s = s.replace("\n", " ")
pf = open("Data/input.txt", "w", encoding='utf-8')
for i in range(100, len(s) - 100, 300):
    answer = ask_question(s[i - 100:i + 400])
    pf.write(answer)

