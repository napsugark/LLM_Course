# import os
# from langfuse import Langfuse
# from langfuse.openai import AzureOpenAI
# from openai import AzureOpenAI as AzureOpenAIUnwrapped
# from dotenv import load_dotenv, find_dotenv

# load_dotenv(find_dotenv())

# # Configure Langfuse
# # It automatically uses the LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and LANGFUSE_HOST env vars.
# langfuse = Langfuse()

# # Configure Azure OpenAI
# # It automatically uses the AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, and AZURE_OPENAI_DEPLOYMENT_NAME env vars.
# azure_client = AzureOpenAI(
#     api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
#     azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
#     api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
#     azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
# )


# def evaluate_and_score_with_langfuse_azure(
#     query: str, context: str, response: str, trace_id: str
# ):
#     """
#     Uses an Azure OpenAI model as an LLM judge, logs the evaluation to Langfuse,
#     and returns the judge's full response.

#     Args:
#         query: The original user query.
#         context: The source text or documents used to generate the response.
#         response: The generated response from your LLM application.
#         trace_id: The ID of the original trace to link the score to.
#     """

#     judge_prompt_messages = [
#         {
#             "role": "system",
#             "content": """
#             You are an expert evaluator. Your task is to judge the correctness of a generated response based *only* on the provided context.

#             Here are the evaluation criteria:
#             1.  **Correctness**: The response must be factually accurate according to the context.
#             2.  **Groundedness**: The response must not contain any information that is not present in the context. If the query cannot be answered by the context, the response should reflect that.

#             Analyze the generated response and provide your evaluation in a structured format.
            
#             Output Format:
#             - **Score**: A numerical score from 1 to 5, where 1 is completely incorrect and 5 is perfectly correct and grounded.
#             - **Reasoning**: A detailed explanation of why you gave that score, pointing out any specific inaccuracies or ungrounded statements.
#             """,
#         },
#         {
#             "role": "user",
#             "content": f"""
#             ---
            
#             **User Query**: {query}
            
#             **Context**:
#             {context}
            
#             **Generated Response**:
#             {response}
            
#             ---
            
#             **Evaluation**:
#             """,
#         },
#     ]

#     # Use a Langfuse trace to log the evaluation
#     with langfuse.trace(name="llm-judge-evaluation", trace_id=trace_id):
#         # Use a Langfuse generation to log the LLM call itself
#         with langfuse.generation(
#             name="context-correctness-judge",
#             model=os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME"),
#             input=judge_prompt_messages,
#         ) as judge_gen:
#             try:
#                 judge_response = azure_client.chat.completions.create(
#                     model=os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME"),
#                     messages=judge_prompt_messages,
#                     temperature=0.0,
#                 )
#                 evaluation_text = judge_response.choices[0].message.content
#                 judge_gen.update(output=evaluation_text)

#                 try:
#                     score_line = next(
#                         line for line in evaluation_text.split("\n") if "Score" in line
#                     )
#                     score = int(score_line.split(":")[1].strip())
#                     reasoning_lines = [
#                         line
#                         for line in evaluation_text.split("\n")
#                         if "Reasoning" in line or not line.strip()
#                     ]
#                     reasoning = (
#                         "\n".join(reasoning_lines[1:]).strip()
#                         if len(reasoning_lines) > 1
#                         else ""
#                     )

#                     langfuse.score(
#                         trace_id=trace_id,
#                         name="context_correctness",
#                         value=score,
#                         comment=reasoning,
#                         observation_id=judge_gen.id,
#                     )
#                     print(f"✅ Evaluation logged to Langfuse for trace ID: {trace_id}")
#                     print(f"Score: {score}")
#                     print(f"Reasoning: {reasoning}")
#                 except (StopIteration, ValueError) as e:
#                     print(f"⚠️ Failed to parse judge output: {e}")
#                     print(f"Full output was:\n{evaluation_text}")
#                     langfuse.score(
#                         trace_id=trace_id,
#                         name="context_correctness",
#                         value=1,
#                         comment=f"Parsing failed: {e}. Raw output: {evaluation_text}",
#                     )
#             except Exception as e:
#                 print(f"❌ An error occurred with Azure OpenAI: {e}")
#                 langfuse.score(
#                     trace_id=trace_id,
#                     name="context_correctness",
#                     value=1,
#                     comment=f"Azure OpenAI call failed: {e}",
#                 )


# # --- Example Usage ---

# # Simulate a trace from your main application (e.g., a RAG pipeline)
# with langfuse.trace(name="rag_response_generation_azure") as trace:
#     # Scenario: Correct and grounded response
#     query_1 = "Who is the inventor of the telephone?"
#     context_1 = "Alexander Graham Bell is widely credited with inventing the telephone."
#     response_1 = "The inventor of the telephone is Alexander Graham Bell."

#     # Log the original generation
#     with trace.generation(
#         name="answer_llm",
#         input=query_1,
#         metadata={"context": context_1},
#         output=response_1,
#     ):
#         pass

#     # Run the judge and score the trace
#     evaluate_and_score_with_langfuse_azure(query_1, context_1, response_1, trace.id)

#     # You can add more scenarios here for batch evaluation.
#     # For example, to evaluate an incorrect response:
#     query_2 = "What is the capital of Japan?"
#     context_2 = (
#         "Kyoto was the former imperial capital of Japan. Tokyo is the current capital."
#     )
#     response_2 = "The capital of Japan is Kyoto."

#     with trace.generation(
#         name="answer_llm",
#         input=query_2,
#         metadata={"context": context_2},
#         output=response_2,
#     ):
#         pass
#     evaluate_and_score_with_langfuse_azure(query_2, context_2, response_2, trace.id)

# # Flush the Langfuse client to ensure all data is sent before the script exits
# langfuse.flush()


import os
from dotenv import load_dotenv, find_dotenv
from langfuse import Langfuse
from langfuse.openai import AzureOpenAI
from openai import AzureOpenAI as AzureOpenAIUnwrapped

# Load environment variables from the .env file
load_dotenv(find_dotenv())

# Configure Langfuse
# It automatically uses the LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and LANGFUSE_HOST env vars.
langfuse = Langfuse()

# Configure Azure OpenAI
# The Langfuse AzureOpenAI client automatically uses the environment variables.
azure_client = AzureOpenAI(
    api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)

# You will need to pass the deployment name to the chat completion call.
# Ensure AZURE_OPENAI_DEPLOYMENT_NAME is set in your .env file.
azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")


def evaluate_and_score_with_langfuse_azure(
    query: str, context: str, response: str, trace_id: str
):
    """
    Uses an Azure OpenAI model as an LLM judge, logs the evaluation to Langfuse.
    ...
    """

    judge_prompt_messages = [
        {
            "role": "system",
            "content": """
            You are an expert evaluator. Your task is to judge the correctness of a generated response based *only* on the provided context.
            ...
            """,
        },
        {
            "role": "user",
            "content": f"""
            ---
            **User Query**: {query}
            **Context**:
            {context}
            **Generated Response**:
            {response}
            ---
            **Evaluation**:
            """,
        },
    ]

    # New API: The trace is created first, then generations are linked to it.
    # The 'llm-judge-evaluation' will be a child of the 'rag-response-generation' trace.
    with langfuse.trace.create(
        name="llm-judge-evaluation", parent_observation_id=trace_id
    ) as judge_trace:
        # The 'model' parameter here in the generation call is crucial for tracking.
        with judge_trace.generation(
            name="context-correctness-judge",
            model=azure_deployment,
            input=judge_prompt_messages,
        ) as judge_gen:
            try:
                judge_response = azure_client.chat.completions.create(
                    model=azure_deployment,
                    messages=judge_prompt_messages,
                    temperature=0.0,
                )
                evaluation_text = judge_response.choices[0].message.content
                judge_gen.update(output=evaluation_text)

                try:
                    score_line = next(
                        line for line in evaluation_text.split("\n") if "Score" in line
                    )
                    score = int(score_line.split(":")[1].strip())
                    reasoning_lines = [
                        line
                        for line in evaluation_text.split("\n")
                        if "Reasoning" in line or not line.strip()
                    ]
                    reasoning = (
                        "\n".join(reasoning_lines[1:]).strip()
                        if len(reasoning_lines) > 1
                        else ""
                    )

                    langfuse.score(
                        trace_id=trace_id,
                        name="context_correctness",
                        value=score,
                        comment=reasoning,
                        observation_id=judge_gen.id,
                    )
                    print(f"✅ Evaluation logged to Langfuse for trace ID: {trace_id}")
                    print(f"Score: {score}")
                    print(f"Reasoning: {reasoning}")
                except (StopIteration, ValueError) as e:
                    langfuse.score(
                        trace_id=trace_id,
                        name="context_correctness",
                        value=1,
                        comment=f"Parsing failed: {e}. Raw output: {evaluation_text}",
                    )
            except Exception as e:
                print(f"❌ An error occurred with Azure OpenAI: {e}")
                langfuse.score(
                    trace_id=trace_id,
                    name="context_correctness",
                    value=1,
                    comment=f"Azure OpenAI call failed: {e}",
                )


# --- Example Usage ---

# The trace for the original RAG generation also needs to use the new API.
with langfuse.trace.create(name="rag-response-generation-azure") as trace:
    # Scenario: Correct and grounded response
    query_1 = "Who is the inventor of the telephone?"
    context_1 = "Alexander Graham Bell is widely credited with inventing the telephone."
    response_1 = "The inventor of the telephone is Alexander Graham Bell."

    # Log the original generation using the new trace object's generation method
    with trace.generation(
        name="answer_llm",
        input=query_1,
        metadata={"context": context_1},
        output=response_1,
    ):
        pass

    # Run the judge and score the trace
    evaluate_and_score_with_langfuse_azure(query_1, context_1, response_1, trace.id)

    # Scenario 2: Incorrect response
    query_2 = "What is the capital of Japan?"
    context_2 = (
        "Kyoto was the former imperial capital of Japan. Tokyo is the current capital."
    )
    response_2 = "The capital of Japan is Kyoto."

    with trace.generation(
        name="answer_llm",
        input=query_2,
        metadata={"context": context_2},
        output=response_2,
    ):
        pass
    evaluate_and_score_with_langfuse_azure(query_2, context_2, response_2, trace.id)

# Flush the Langfuse client to ensure all data is sent before the script exits
langfuse.flush()