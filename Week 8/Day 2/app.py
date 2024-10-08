import os
from dotenv import load_dotenv
import chainlit as cl
from langchain_community.llms import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from operator import itemgetter
from langchain.schema.runnable import RunnablePassthrough

# Load environment variables
load_dotenv()

# Set up HuggingFace LLM
YOUR_LLM_ENDPOINT_URL = "https://eqja61s66z7140xs.us-east-1.aws.endpoints.huggingface.cloud"
hf_llm = HuggingFaceEndpoint(
    endpoint_url=YOUR_LLM_ENDPOINT_URL,
    max_new_tokens=512,
    top_k=10,
    top_p=0.95,
    typical_p=0.95,
    temperature=0.01,
    repetition_penalty=1.03,
    huggingfacehub_api_token=os.environ["HF_TOKEN"]
)

# Set up RAG prompt template
RAG_PROMPT_TEMPLATE = """\
<|start_header_id|>system<|end_header_id|>
You are a helpful assistant. You answer user questions based on provided context. If you can't answer the question with the provided context, say you don't know.<|eot_id|>

<|start_header_id|>user<|end_header_id|>
User Query:
{query}

Context:
{context}<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
"""
rag_prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

# Set up HuggingFace Embeddings
YOUR_EMBED_MODEL_URL = "https://y7nlf8p9ats9hfsv.us-east-1.aws.endpoints.huggingface.cloud"
hf_embeddings = HuggingFaceEndpointEmbeddings(
    model=YOUR_EMBED_MODEL_URL,
    task="feature-extraction",
    huggingfacehub_api_token=os.environ["HF_TOKEN"],
)

# Load and prepare documents
document_loader = TextLoader("./paul-graham-to-kindle/paul_graham_essays.txt")
documents = document_loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=30)
split_documents = text_splitter.split_documents(documents)

# Create FAISS vectorstore
vectorstore = FAISS.from_documents(split_documents[:32], hf_embeddings)
for i in range(32, len(split_documents), 32):
    vectorstore.add_documents(split_documents[i:i+32])

hf_retriever = vectorstore.as_retriever()

# Set up LCEL RAG chain
lcel_rag_chain = (
    {"context": itemgetter("query") | hf_retriever, "query": itemgetter("query")}
    | rag_prompt
    | hf_llm
)

@cl.on_chat_start
def start():
    cl.user_session.set("chain", lcel_rag_chain)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    res = await cl.make_async(chain.invoke)({"query": message.content})
    await cl.Message(content=res).send()

if __name__ == "__main__":
    cl.run()