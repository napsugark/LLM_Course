from openai import AzureOpenAI
import chainlit as cl
import os
from dotenv import find_dotenv, load_dotenv
from chainlit.input_widget import Select, Slider

load_dotenv(find_dotenv())


@cl.on_chat_start
async def start_chat():
    # Set up settings with the Select widget
    await cl.ChatSettings(
        [
            Select(
                id="chatbot_language",
                label="Selectați limba chatbot-ului",
                values=["ro", "en", "fr"],
                initial_index=0,
                tooltip="Alege limba în care chatbot-ul va răspunde",
            ),
            Slider(
                id="temperature",
                label="Temperatura",
                initial=0.7,
                min=0.0,
                max=2.0,
                step=0.1,
                tooltip="Controlează creativitatea răspunsurilor (0.0 = conservator, 2.0 = foarte creativ)",
            ),
        ]
    ).send()

    # Set default language from initial_index
    cl.user_session.set("language", "ro")  # Default to first option
    cl.user_session.set("temperature", 0.7)

    cl.user_session.set(
        "client",
        AzureOpenAI(
            azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
            api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
            api_version="2024-06-01",
        ),
    )
    cl.user_session.set("chat_history", [])

    language = cl.user_session.get("language", "ro")
    language_labels = {"ro": "română", "en": "engleză", "fr": "franceză"}

    await cl.Message(
        content=f"👋 Bun venit! Întreabă-mă orice și voi răspunde în {language_labels[language]}. Poți schimba limba din setări.",
        actions=[
            cl.Action(
                name="reset",
                value="reset",
                label="🔄 Resetează conversația",
                payload={},
            ),
        ],
    ).send()


# Add this callback to handle language selection
@cl.on_settings_update
async def setup_agent(settings):
    language = settings["chatbot_language"]
    temperature = settings["temperature"]

    cl.user_session.set("language", language)
    cl.user_session.set("temperature", temperature)

    language_labels = {"ro": "română", "en": "engleză", "fr": "franceză"}
    await cl.Message(
        content=f"✅ Limba a fost schimbată în {language_labels[language]}."
    ).send()


@cl.on_message
async def handle_message(message: cl.Message):
    msg = cl.Message(content="")
    language = cl.user_session.get("language", "ro")
    temperature = cl.user_session.get("temperature", 0.7)
    system_prompt = {
        "ro": "Răspunde în limba română.",
        "en": "Respond in English.",
        "fr": "Réponds en français.",
    }

    chat_history = cl.user_session.get("chat_history", [])
    chat_history.append({"role": "user", "content": f"QUESTION: {message.content}"})

    response = cl.user_session.get("client").chat.completions.create(
        model="gpt-4o-mini",
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt[language]},
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


# Handle reset button
@cl.action_callback("reset")
async def on_reset(action: cl.Action):
    cl.user_session.set("chat_history", [])
    await cl.Message(content="✅ Conversația a fost resetată. Întreabă din nou!").send()

