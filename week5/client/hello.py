# /// script
# requires-python = ">=3.11"
# dependencies = ["openai"]
# ///
"""Тот же запрос, что curl с client/chat.json, из Python: любой клиент OpenAI API ходит в vLLM без правок.

  uv run client/hello.py        из папки week5, туннель открыт (bash laptop/tunnel.sh)
"""
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
r = client.chat.completions.create(
    model="base", max_tokens=40,
    messages=[{"role": "user", "content": "One sentence about retrospot lunch bags."}])
print(r.choices[0].message.content)
print(r.usage)
