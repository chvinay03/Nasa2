import os
from typing import Dict, List
from openai import OpenAI

def generate_response(openai_key: str, user_message: str, context: str,
                     conversation_history: List[Dict], model: str = "gpt-3.5-turbo") -> str:
    """Generate a grounded response using OpenAI with NASA document context."""

    api_key = openai_key or os.environ.get("OPENAI_API_KEY")

    mission_specialist_prompt = """You are an expert NASA mission specialist with deep knowledge of historic
space missions including Apollo 11, Apollo 13, and the Challenger disaster (STS-51L).
Answer questions based ONLY on the provided context from official NASA documents.
Always cite the source when referencing information.
If the context does not contain enough information, clearly say so.
Do not speculate or add information not present in the context."""

    chat_messages = [{"role": "system", "content": mission_specialist_prompt}]

    if context:
        chat_messages.append({
            "role": "user",
            "content": f"Here is the relevant context from NASA documents:\n\n{context}\n\nPlease use this to answer my questions."
        })
        chat_messages.append({
            "role": "assistant",
            "content": "Understood. I have reviewed the NASA documents and will answer based on this context."
        })

    for msg in conversation_history:
        if msg.get("role") in ["user", "assistant"]:
            chat_messages.append({"role": msg["role"], "content": msg["content"]})

    chat_messages.append({"role": "user", "content": user_message})

    client = OpenAI(api_key=api_key, base_url="https://openai.vocareum.com/v1")

    response = client.chat.completions.create(
        model=model,
        messages=chat_messages,
        temperature=0.3,
        max_tokens=1000
    )

    return response.choices[0].message.content
