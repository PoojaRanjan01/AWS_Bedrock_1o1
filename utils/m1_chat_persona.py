"""
Bedrock chat with Travel Persona
"""

from langchain.chains import ConversationChain
from langchain.llms.bedrock import Bedrock
from langchain.memory import ConversationBufferMemory

import boto3
bedrock_runtime_client = boto3.client(service_name="bedrock-runtime")

memory = ConversationBufferMemory()
memory.human_prefix = "User"
memory.ai_prefix = "Bot"
memory.chat_memory.add_user_message("You will be acting as a travel guide."
                                    "You have all the knowledge about all tourist places in the world.")
memory.chat_memory.add_ai_message("I am world travel guide and I give answers to any travel query.")


def get_bedrock_llm(selected_llm):
    """
    Initialize LLM model

    :param selected_llm: LLM type
    """
    print(f"[INFO] Selected LLM is : {selected_llm}")
    if selected_llm in ['anthropic.claude-v2', 'anthropic.claude-v1', 'anthropic.claude-instant-v1']:
        llm = Bedrock(model_id=selected_llm, model_kwargs={'max_tokens_to_sample': 4096})

    elif selected_llm in ['amazon.titan-tg1-large', 'amazon.titan-text-express-v1', 'amazon.titan-text-lite-v1']:
        llm = Bedrock(
            model_id=selected_llm,
            model_kwargs={
                "maxTokenCount": 4096,
                "stopSequences": [],
                "temperature": 0,
                "topP": 1,
            }
        )
    else:
        raise ValueError(f"Unsupported LLM: {selected_llm}")

    return llm

def llm_conversation_model(selected_llm):
    """
    Initialize LLM model conversation chain
    :param selected_llm: LLM type
    """
    llm = get_bedrock_llm(selected_llm)
    conversation_chain = ConversationChain(
        llm=llm, verbose=True, memory=memory
    )
    return conversation_chain


def AI_chat_responder(travel_query, conversation_chain):
    """
    Respond to user query through LLM model
    :param travel_query: user query
    :param conversation_chain: LLM message chain
    """
    query_suffix = " Please tell in short and only answer to this question, no further."
    Res = conversation_chain.predict(input=travel_query + query_suffix)
    Res_display = Res.split('\nHuman')[0]

    return Res_display
