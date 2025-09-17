from langfuse.openai import AzureOpenAI
from langchain_openai import AzureOpenAIEmbeddings
import chainlit as cl
import os
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from dotenv import load_dotenv, find_dotenv
from chainlit.input_widget import Select, Slider
import uuid
from langfuse import observe, get_client

load_dotenv(find_dotenv())


@cl.on_chat_start
async def start_chat():
    """Initialize chat session with unique IDs and components"""
    # Generate unique identifiers
    session_id = str(uuid.uuid4())
    user_id = str(uuid.uuid4())

    # Store session info
    cl.user_session.set("session_id", session_id)
    cl.user_session.set("user_id", user_id)
    cl.user_session.set("chat_history", [])

    # Initialize chat settings
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

    # Initialize Azure OpenAI client with Langfuse tracing
    azure_client = AzureOpenAI(
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
        api_version="2024-06-01",
    )
    cl.user_session.set("client", azure_client)

    # Initialize embedding model
    cl.user_session.set(
        "embedding_model",
        AzureOpenAIEmbeddings(
            model="text-embedding-ada-002",
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-06-01",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        ),
    )

    # Initialize Qdrant client and retriever
    qdrant_client = QdrantClient(
        url=os.getenv("QDRANT_ENDPOINT"), api_key=os.getenv("QDRANT_API_KEY")
    )
    cl.user_session.set("qdrant_client", qdrant_client)

    retriever = QdrantVectorStore(
        collection_name="local_movie_db",
        embedding=cl.user_session.get("embedding_model"),
        client=qdrant_client,
    )
    cl.user_session.set("retriever", retriever)


def get_system_prompt(language: str) -> str:
    """Generate system prompt based on selected language"""
    return f"""
You are a helpful assistant for questions about famous movies.
You will formulate all your answers in {language}.
Base your answers only on pieces of information received as context below.
If you don't know the answer, just say that you don't know.
Do not answer any question that is not related to movies."""


@cl.on_settings_update
async def setup_agent(settings):
    """Update user session settings when user changes them"""
    cl.user_session.set("language", settings["language"])
    cl.user_session.set("temperature", settings["Temperature"])


def format_docs(docs) -> str:
    """Format retrieved documents into a single string"""
    return "\n\n".join(doc.page_content for doc in docs)


@observe(name="movie_chat_message")
async def process_movie_query(message_content: str, language: str, temperature: float):
    """Process movie-related query with RAG pipeline"""

    # Get the Langfuse client and update current trace with session_id
    langfuse = get_client()
    langfuse.update_current_trace(
        session_id=cl.user_session.get("session_id"),
        user_id=cl.user_session.get("user_id"),
        metadata={
            "language": language,
            "temperature": temperature,
        },
    )
    retriever = cl.user_session.get("retriever")
    client = cl.user_session.get("client")

    # Retrieve relevant documents
    retrieved_docs = retriever.similarity_search(message_content, k=4)
    context = format_docs(retrieved_docs)

    # Prepare messages for the chat completion
    system_prompt = get_system_prompt(language)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"QUESTION: {message_content}"},
        {"role": "system", "content": f"CONTEXT: {context}"},
    ]

    # Generate response with streaming
    full_response = ""
    msg = cl.Message(content="")
    await msg.send()

    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=temperature,
        stream=True,
        messages=messages,
        user=cl.user_session.get("user_id"),
    )

    for chunk in stream:
        if not chunk.choices or not chunk.choices[0].delta:
            continue
        delta = chunk.choices[0].delta.content or ""
        full_response += delta
        await msg.stream_token(delta)

    await msg.update()

    # Add context as side element
    source_elements = [cl.Text(content=context, name="Context", display="side")]
    msg.elements = source_elements
    await msg.update()

    return full_response, context


@observe(name="collect_feedback")
async def collect_user_feedback():
    """Collect user feedback for the response"""

    # Update current trace with session info
    langfuse = get_client()
    langfuse.update_current_trace(
        session_id=cl.user_session.get("session_id"),
        user_id=cl.user_session.get("user_id"),
    )
    feedback_msg = await cl.AskActionMessage(
        content="Was this response helpful?",
        actions=[
            cl.Action(name="feedback", payload={"value": 1}, label="👍"),
            cl.Action(name="feedback", payload={"value": 0}, label="👎"),
        ],
    ).send()

    if feedback_msg:
        feedback_value = feedback_msg.get("payload", {}).get("value", 0)
        await cl.Message(content="Thank you for your feedback!").send()
        return feedback_value
    return None


@cl.on_message
@observe(name="chat_session")
async def message_handler(message: cl.Message):
    """Main message handler with proper Langfuse session tracking"""

    # Get the Langfuse client and update current trace with session_id
    langfuse = get_client()
    langfuse.update_current_trace(
        session_id=cl.user_session.get("session_id"),
        user_id=cl.user_session.get("user_id"),
        metadata={
            "message_content": message.content,
        },
    )
    # Get user settings
    language = cl.user_session.get("language", "English")
    temperature = cl.user_session.get("temperature", 0.0)
    chat_history = cl.user_session.get("chat_history", [])
    session_id = cl.user_session.get("session_id")
    user_id = cl.user_session.get("user_id")

    # Process the movie query
    try:
        response, context = await process_movie_query(
            message.content, language, temperature
        )

        # Update chat history
        chat_history.extend(
            [
                {
                    "role": "user",
                    "content": message.content,
                    "timestamp": str(
                        uuid.uuid4()
                    ),  # You might want to use actual timestamps
                },
                {
                    "role": "assistant",
                    "content": response,
                    "context": context,
                    "timestamp": str(uuid.uuid4()),
                },
            ]
        )
        cl.user_session.set("chat_history", chat_history)

        # Collect feedback after a few exchanges
        if len(chat_history) > 2:
            feedback = await collect_user_feedback()

    except Exception as e:
        error_msg = f"An error occurred while processing your request: {str(e)}"
        await cl.Message(content=error_msg).send()
        raise


# Optional: Add a function to retrieve chat history for debugging
@observe(name="get_chat_history")
def get_session_history(session_id: str):
    """Retrieve chat history for a specific session (utility function)"""

    # Update current trace with session info
    langfuse = get_client()
    langfuse.update_current_trace(
        session_id=session_id, metadata={"action": "retrieve_history"}
    )

    # This would typically interact with your database or storage
    # For now, it just returns the current session history
    return cl.user_session.get("chat_history", [])