import os
import boto3

from langchain.llms.bedrock import Bedrock
from langchain.memory import ConversationBufferMemory

from langchain.embeddings import BedrockEmbeddings
from langchain.vectorstores import FAISS
from langchain.indexes.vectorstore import VectorStoreIndexWrapper

bedrock_runtime_client = boto3.client(service_name="bedrock-runtime")

# embedding model
br_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_runtime_client)
# flag_reembed = False
flag_reembed = True

# # LLM
modelId = "amazon.titan-tg1-large"
titan_llm = Bedrock(model_id=modelId, client=bedrock_runtime_client)
titan_llm.model_kwargs = {'temperature': 0.5, "maxTokenCount": 700}

memory = ConversationBufferMemory()

from langchain.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter

def load_data_file(csv_file_path):
    """
    Load csv data file from local and split into chunks

    :param csv_file_path: local file path
    """
    loader = CSVLoader(csv_file_path)
    documents_aws = loader.load()
    print(f"documents:loaded:size={len(documents_aws)}")

    docs = CharacterTextSplitter(chunk_size=2000, chunk_overlap=400, separator=",").split_documents(documents_aws)
    print(f"Documents:after split and chunking size={len(docs)}")
    return docs


def get_data_embeddings(docs):
    """
    Embed data file and store as Faiss vectorstore in local

    :param docs: document files
    """
    model_dir = 'models'
    embedding_path = model_dir + '/faiss_vector_store'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if flag_reembed:
        vectorstore_faiss_aws = FAISS.from_documents(
            documents=docs,
            embedding=br_embeddings)
        # Storing faiss model
        vectorstore_faiss_aws.save_local(embedding_path)
    else:
        if os.path.exists(embedding_path):
            # Loading faiss model
            vectorstore_faiss_aws = FAISS.load_local(embedding_path, br_embeddings)
        else:
            # Training Faiss embeddings
            vectorstore_faiss_aws = FAISS.from_documents(
                documents=docs,
                embedding=br_embeddings)
            # Storing faiss model
            vectorstore_faiss_aws.save_local(embedding_path)
    return vectorstore_faiss_aws

def llm_context_setup(csv_file_path):
    """
    Read local data file

    :param csv_file_path: path of file
    """
    docs = load_data_file(csv_file_path)

    vectorstore_faiss_aws = get_data_embeddings(docs)
    print(f"vectorstore_faiss_aws:created={vectorstore_faiss_aws}::")
    return vectorstore_faiss_aws


def llm_context_query(user_question, vectorstore_faiss_aws):
    """
    Query on Faiss VectorStore database

    :param user_question: Question asked by user
    :param vectorstore_faiss_aws: vectorstore object
    """

    wrapper_store_faiss = VectorStoreIndexWrapper(vectorstore=vectorstore_faiss_aws)
    llm_response = wrapper_store_faiss.query(user_question,
                                             llm=titan_llm)
    return llm_response

