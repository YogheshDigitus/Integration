from langchain.chains import RetrievalQA
from sklearn.metrics.pairwise import cosine_similarity
from langchain_ollama.llms import OllamaLLM
import re
# from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from langchain.schema import Document
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from openai import OpenAI
from langchain.chains import LLMChain
import torch
from transformers import CLIPModel, CLIPProcessor
from langchain.embeddings.base import Embeddings
from PIL import Image
import numpy as np
import chromadb
# from langchain.retrievers import VectorstoreRetriever
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from dotenv import load_dotenv

load_dotenv()
from langchain.globals import set_debug
from langchain_openai import ChatOpenAI
from operator import itemgetter
from langchain.retrievers import (ContextualCompressionRetriever, MergerRetriever, )

# DocumentCompressorPipeline


set_debug(True)

import openai
import os

client = OpenAI()
chroma_client = chromadb.PersistentClient(path=r"C:\Users\DELL\Desktop\Chatbot\My_chat_bot\VectorDB")# can also use server local
# Define your API key
# openai.api_key = 'your_openai_api_key'


from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import openai
from langchain.retrievers import MultiQueryRetriever

import pprint

# Define the custom prompt template
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")  # CLIP for both text and image embeddings
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")  # CLIP processor for images

# Initialize the OpenAI LLM
class CLIPEmbeddings(Embeddings):
    def __init__(self):
        self.model = clip_model
        self.processor = clip_processor

    def embed_text(self, texts):
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        with torch.no_grad():
            text_embeddings = self.model.get_text_features(**inputs)
        return text_embeddings.numpy()

    def embed_documents(self, texts):
        return self.embed_text(texts)

    def embed_query(self, query):
        return self.embed_text([query])[0]

    def embed_image(self, image_path):
        image = Image.open(image_path)
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            image_embeddings = self.model.get_image_features(**inputs)
        return image_embeddings.numpy()

    def embed_images(self, image_paths):
        return [self.embed_image(path) for path in image_paths]
def is_requesting_image(user_query):
    image_keywords = ["image", "picture", "photo", "visual", "screenshot", "graphic", "diagram", "chart", "illustration", "pic"]
    action_keywords = ["show me", "display", "draw", "see", "view", "render"]

    user_query = user_query.lower()

    # Check if any image-related or action-related keywords are in the query
    if any(keyword in user_query for keyword in image_keywords) or any(action in user_query for action in action_keywords):
        return True
    else:
        return False

def rag_pipeline_with_prompt(query):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    # Define your prompt template
    template = """
    You are a help desk technician called Lilis from Linxens company. You are interacting with a user who is asking you questions about the company's issues. Based on the following user question and context provided, please give detailed answer to the user question.
    don't give any thing apart from answer.
    Don't include irrelavent information like legal disclaimers, proprietary information, or structural like page number, index etc.
    They to be as comprehensive as possible giving well rounded answer
    If you don't know the answer or it is not present in context provided, just say that you don't know, don't try to 
    make up an answer.
    \n
    chat_history previous three conversation.
    \n\n
    Question: {question}
    Context: {context}

    """
    prompt_final = PromptTemplate(
        template=template,
        input_variables=["chat_history", "context", "question"]
    )

    # Load ChromaDB client
    persist_directory = r"C:\Users\DELL\Desktop\Chatbot\My_chat_bot\VectorDB"
    chroma_client = chromadb.PersistentClient(persist_directory)
    clip_embeddings = CLIPEmbeddings()
    # Retrieve the collection that contains your embeddings (assuming it's stored under a name like "documents")
    collection = chroma_client.get_collection("multimodel_collection_1",embedding_function=OpenCLIPEmbeddingFunction())
    query_emb=clip_embeddings.embed_text([query])
    print(f" \n Query for the collection is {query}\n")
    query_result = collection.query(
    query_texts=query,
    include=["embeddings","distances","documents","metadatas"],
    n_results=5,
    )
    documents = query_result["documents"]
    print(query_result)
    distances = query_result["distances"]
    image_path_dic=query_result["metadatas"][0]
    print(image_path_dic)
    image_path=[]
    for img in image_path_dic:
        if 'image' in img:        
            image_path=image_path+[img['image']]

#----------------------------------------Commenting out the image retrieval part-------------------------------#
    # if is_requesting_image(query)==True:
    #     query_result_image = collection.query(
    #     query_embeddings=query_emb,
    #     include=["embeddings","distances","documents","metadatas"],
    #     n_results=2,
    #     )
    #     metadatas_image = query_result_image["metadatas"]
    #     ids_images=(query_result_image["ids"])[0]
    #     pattern = r"(?<=_)(extracted_images\\.*)"
    #     for i,metadata in enumerate(metadatas_image[0]):
    # # Check if 'type' key exists and if its value is 'image'
    #         if "type" in metadata and metadata["type"] == "image":
    #             match = re.search(pattern, ids_images[i])
    #             image_path.append(match.group(1))
    #         else:
    #             print(None)
#----------------------------------------Commenting out the image retrieval part-------------------------------#

    context = "\n".join(documents[0])  # Joining the top documents as the context for the model  # Optionally append image info to the context (or you can process them separately)
    rag_chain = (
        {
            "context": itemgetter("context"),
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history"),
        }
        | prompt_final  # Pass the custom prompt into the chain
        | llm  # Use the language model for answering
        | StrOutputParser()  # Parse the output
    )
    print(f"context of the query is {context}")
    result = rag_chain.invoke({"question": query,"context":context})
    return result # for testing image is not included

def Get_summary(context):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    # # Now, prepare the retriever if necessary (same as before)
    template = """
    Your are providing the summary of the document the user has upload.
    Generate a concise and engaging summary of the provided document as context, focusing on:
        A brief introduction to the subject or product.
        Key topics or features covered in the document with the specifics.
        A closing statement to encourage users to explore or use the content
    Don't include irrelavent information like legal disclaimers, proprietary information, or structural like page number, index etc.  
    If you don't know the answer or it is not present in context provided, just say that you don't know, don't try to 
    make up an answer.
    Do not make assumptions or provide information beyond the given context.
    Context: {context}"""
    prompt_final = PromptTemplate(
        template=template,
        input_variables=["context"]
    )
    rag_chain = (
        {
            "context": itemgetter("context"),
        }
        | prompt_final  # Pass the custom prompt into the chain
        | llm  # Use the language model for answering
        | StrOutputParser()  # Parse the output
    )
    # MergerRetriever can be used if you're combining multiple retrievers
    # # Execute the RAG pipeline with the user query and chat history
    result = rag_chain.invoke({"context":context})

    return result
