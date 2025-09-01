from openai import AzureOpenAI
import chainlit as cl
import os
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())


@cl.on_chat_start
async def start_chat():
    cl.user_session.set(
        "client",
        AzureOpenAI(
            azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
            api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
            api_version="2024-06-01",
        ),
    )
    cl.user_session.set("chat_history", [])


@cl.on_message
async def handle_message(message: cl.Message):
    msg = cl.Message(content="")

    chat_history = cl.user_session.get("chat_history", [])
    chat_history.append({"role": "user", "content": f"QUESTION: {message.content}"})

    response = cl.user_session.get("client").chat.completions.create(
        model="gpt-4o-mini",
        # messages=[{"role": m["role"], "content": m["content"]} for m in chat_history],
        messages=[
            {
                "role": "system",
                "content": "Răspunde întotdeauna în limba română, indiferent de întrebarea utilizatorului.",
            },
            *[{"role": m["role"], "content": m["content"]} for m in chat_history],
        ],
        stream=True,
    )

    for chunk in response:
        if not chunk.choices or not chunk.choices[0].delta:
            continue
        delta = chunk.choices[0].delta.content or ""
        await msg.stream_token(delta)

    # Update chat history
    chat_history.append({"role": "user", "content": message.content})
    chat_history.append({"role": "assistant", "content": msg.content})
    cl.user_session.set("chat_history", chat_history)
    await msg.update()
