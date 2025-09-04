from openai import AzureOpenAI
from langchain_openai import AzureOpenAIEmbeddings
import chainlit as cl
import os
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from dotenv import load_dotenv, find_dotenv
from chainlit.input_widget import Select, Slider


# SCENARIO="original" # example from notebook
global SCENARIO
SCENARIO = "assignment_1"  #  1st assignment

load_dotenv(find_dotenv())


@cl.on_chat_start
async def start_chat():
    await cl.ChatSettings(
        [
            Select(
                id="language",
                values=["English", "Romanian"],
                label="Select your preferred language",
                initial_value="English",
            ),
            Slider(
                id="Temperature", label="Temperature", initial=0, min=0, max=1, step=0.1
            ),
        ]
    ).send()
    cl.user_session.set("chat_history", [])
    cl.user_session.set(
        "client",
        AzureOpenAI(
            azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
            api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
            api_version="2024-06-01",
        ),
    )
    cl.user_session.set(
        "embedding_model",
        AzureOpenAIEmbeddings(
            model="text-embedding-ada-002",
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-06-01",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        ),
    )
    cl.user_session.set(
        "qdrant_client",
        QdrantClient(
            url=os.getenv("QDRANT_ENDPOINT"), api_key=os.getenv("QDRANT_API_KEY")
        ),
    )
    cl.user_session.set(
        "retriever",
        QdrantVectorStore(
            collection_name="local_movie_db",
            embedding=cl.user_session.get("embedding_model"),
            client=cl.user_session.get("qdrant_client"),
        ),
    )


def get_system_prompt(language):
    return f"""
    You are a helpful assistant for question on famous movies.
    You will formulate all its answers in {language}.
    Base you answer only on pieces of information received as context below.
    If you don't know the answer, just say that you don't know.
    Do not answer any question that are not related to movies."""


@cl.on_settings_update
async def setup_agent(settings):
    cl.user_session.set("language", settings["language"])
    cl.user_session.set("temperature", settings["Temperature"])
    if SCENARIO == "assignment_1":
        embedding_model = cl.user_session.get("embedding_model")
        qdrant_client = cl.user_session.get("qdrant_client")
        retriever = get_retriever(settings["language"], embedding_model, qdrant_client)
        cl.user_session.set("retriever", retriever)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_retriever(language, embedding_model, qdrant_client):
    collection_name = (
        "local_movie_db_en" if language == "English" else "local_movie_db_ro"
    )
    return QdrantVectorStore(
        collection_name=collection_name, embedding=embedding_model, client=qdrant_client
    )

async def reformulate_question(client, chat_history, latest_question, language):
    """
    Reformulate follow-up questions into standalone questions
    using only user questions + assistant responses.
    """
    conversation_turns = [m for m in chat_history if m["role"] in ["user", "assistant"]]

    reformulation_prompt = [
        {
            "role": "system",
            "content": f"You are a helpful assistant. Reformulate follow-up questions into fully standalone questions in {language}.",
        },
        *conversation_turns,
        {
            "role": "user",
            "content": f"Rewrite this question so it is self-contained: {latest_question}",
        },
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini", temperature=0, messages=reformulation_prompt
    )

    reformulated = response.choices[0].message.content.strip()
    return latest_question, reformulated


@cl.on_message
async def message_send(message: cl.Message):
    language = cl.user_session.get("language", "English")
    temperature = cl.user_session.get("temperature", 0)
    chat_history = cl.user_session.get("chat_history", [])
    retriever = cl.user_session.get("retriever")
    client = cl.user_session.get("client")

    original_question, reformulated_question = await reformulate_question(
        client, chat_history, message.content, language
    )

    retrieved_docs = retriever.similarity_search(reformulated_question, k=4)
    context = format_docs(retrieved_docs)

    system_prompt = get_system_prompt(language)
    chat_history.append({"role": "system", "content": system_prompt})
    chat_history.append(
        {"role": "user", "content": f"QUESTION: {reformulated_question}"}
    )
    chat_history.append({"role": "system", "content": f"CONTEXT: {context}"})


    full_response = ""
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=temperature,
        stream=True,
        messages=[{"role": m["role"], "content": m["content"]} for m in chat_history],
    )

    msg = cl.Message(content="")
    await msg.send()
    for chunk in stream:
        if not chunk.choices or not chunk.choices[0].delta:
            continue
        delta = chunk.choices[0].delta.content or ""
        full_response += delta
        await msg.stream_token(delta)
    await msg.update()

    # debug_elements = [
    #     cl.Text(
    #         content=f"Original: {original_question}",
    #         name="Original Question",
    #         display="side",
    #     ),
    #     cl.Text(
    #         content=f"Reformulated: {reformulated_question}",
    #         name="Reformulated Question",
    #         display="side",
    #     ),
    #     cl.Text(content=context, name="Context", display="side"),
    # ]
    # msg.elements = debug_elements

    debug_content = (
    f"**Original Question:** {original_question}\n\n"
    f"**Reformulated Question:** {reformulated_question}\n\n"
    f"**Retrieved Context:**\n{context}"
    )

    # Create the chat message with assistant response + debug info
    msg = cl.Message(content=full_response + "\n\n" + debug_content)
    print("Message content:\n \n", msg.content)
    await msg.update()

    chat_history.append({"role": "assistant", "content": full_response})
    cl.user_session.set("chat_history", chat_history)
