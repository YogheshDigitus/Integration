# new package to add to requirement are scikit-learn, requests, pdfplumber
import os
import math
import chromadb.api
import pythoncom
import chromadb
import numpy as np
from docx2pdf import convert
import fitz  # PyMuPDF
from io import BytesIO
from PIL import Image
from docx import Document as DocxDocument
from docx.shared import Inches
from transformers import CLIPProcessor, CLIPModel
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
import torch
from langchain_community.vectorstores.utils import filter_complex_metadata
from docx import Document as DocxDocument
from docx.shared import Inches
# import pytesseract
# import openai
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.docstore.document import Document
from openai import OpenAI
# import json
from dotenv import load_dotenv
# import requests

#setting up the OpenAI API key 
#chroma_client = chromadb.PersistentClient(path=r"C:\Users\DELL\Desktop\Chatbot\My_chat_bot\VectorDB")# can also use server local
#path= r"C:\Users\DELL\Desktop\Chatbot\My_chat_bot\Transformers"
# Initialize CLIP model and processor from Hugging Face

path= r"C:\Users\DELL\Desktop\Chatbot\My_chat_bot\Transformers"
clip_model = CLIPModel.from_pretrained(path)
clip_processor = CLIPProcessor.from_pretrained(path)
device = "cpu"
clip_model.to(device)



class CSVProcessor:
    def __init__(self, file_path, persist_directory='persist_chroma_csv'):
        self.file_path = file_path
        self.persist_directory = persist_directory
        self.loader = CSVLoader(file_path=self.file_path, encoding='utf-8', csv_args={'delimiter': '|'})
        self.embedding = OpenAIEmbeddings()

    def process(self, filename):
        self.subject = filename
        docs = self.loader.load()
        # vectordb = Chroma.from_documents(
        #     documents=docs,
        #     embedding=self.embedding,
        #     persist_directory=self.persist_directory
        # )
        # vectordb.persist()

class PDFProcessor:
    def __init__(self, file_path, file_extension,persist_directory='persist_chroma_pdf'):
        self.file_path = file_path
        self.file_extension=file_extension
        self.persist_directory = persist_directory
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")  # CLIP for both text and image embeddings
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")  # CLIP processor for images
        self.subject = None
        self.device= "cpu"

    def process(self, filename):
        """Process the PDF, extract text, images, and store them in Chroma."""
        self.subject = filename
        doc = fitz.open(self.file_path)
        chromadb.api.client.SharedSystemClient.clear_system_cache()
        output_image_dir = 'extracted_images'
        text_embeddings = []
        Document=[]
        image_embeddings = []
        image_paths=[]
        image_arrays=[]
        embed_func=OpenCLIPEmbeddingFunction()
        data_loader = ImageLoader()

        #Below path should be changed based on the user case
        output_text_file_path=fr"C:\Users\DELL\Desktop\Chatbot\My_chat_bot\Summary\{filename}{self.file_extension}.text" 
        text_file = open(output_text_file_path, 'w',encoding='utf-8')
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("text")
            text_file.write(f"Page {page_num + 1}:\n{text}\n\n")
            if not os.path.exists(output_image_dir):
                os.makedirs(output_image_dir)
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_data = base_image["image"]

            # Convert image data to a Pillow Image object
                image = Image.open(BytesIO(image_data))

            # Generate image filename based on page and image index
                image_filename = f"{filename}_page_{page_num + 1}_img_{img_index + 1}.jpg"
                image_path = os.path.join(output_image_dir, image_filename)

            # Save the image to the specified directory
                if image.mode == 'RGBA' or 'P' or '1':
            # Convert to RGB
                    image = image.convert('RGB')
                image.save(image_path)
                img_array = np.array(image)
                image_arrays.append(img_array)
                image_paths.append(image_path)
        text_file.close()
        chunks=self.Text_chunker(output_text_file_path)

        # Persistant path should also be changed based on the use case
        persist_directory= r"C:\Users\DELL\Desktop\Chatbot\My_chat_bot\VectorDB"
        Chroma_client = chromadb.PersistentClient(persist_directory)
        collection3=Chroma_client.get_or_create_collection(
        name='multimodel_collection_1',
        embedding_function=embed_func,
        metadata={
            "hnsw:space": "cosine",
            "hnsw:search_ef": 100
        },
        data_loader=data_loader)
        Summary_collection=Chroma_client.get_or_create_collection(name='Summary_collection',embedding_function=embed_func)
        ids_text=[f"{filename}_text{id+1}" for id in range(len(chunks))]
        collection3.add(
            documents=chunks,
            ids=ids_text,
            metadatas=[{'subject': self.subject,"type": "text", "file_type":"PDF"} for i in range(len(ids_text))]
        )
        if image_paths:
            ids_img=[f"{os.path.splitext(os.path.basename(self.file_path))[0]}_{id}" for id in image_paths]
            collection3.add(
                ids=ids_img,
                images=image_arrays,
                metadatas=[{"type":"image"} for i in range(len(ids_img))]
            )

 # the function return the text and the image path where the extracted images are saved   



###########-------Not used at the moment---------------###############
#     def get_text_embedding(self,chunks):
#         Text_embeddings = []

# # Iterate over each text chunk
#         for chunk in chunks:
#             # Preprocess the text and convert to embeddings
#             inputs = clip_processor(text=chunk, return_tensors="pt", padding=True, truncation=True)

#             # Forward pass through the model
#             with torch.no_grad():
#                 outputs = clip_model.get_text_features(**inputs)

#             # Extract the embeddings (usually the last hidden state or pooled output)
#             Text_embeddings.append(outputs.flatten().tolist())
#         return Text_embeddings



####################----------------not used at the moment---------------##########################
    # def get_image_embeddings(self, image_paths):
    #     image_embeddings = []
    #     os  # List to store (embedding, path) pairs
    #     for image_path in image_paths:
    #         # Open the image file from disk
    #         image = Image.open(image_path)

    #         # Preprocess the image and get embeddings
    #         inputs = clip_processor(images=image, return_tensors="pt")
    #         with torch.no_grad():
    #             image_embedding = clip_model.get_image_features(**inputs)

    #         # Store the embedding along with the path as a tuple
    #         image_embeddings.append(image_embedding.flatten().tolist())

    #     return image_embeddings
    
    def summarize_text(self, name, text, file_type="pdf", vectordb="persist_chroma_pdf"):
        # self.summarize_summary(text)
        document = {}
        document["name"] = name
        document["summary"] = self.summarize_summary(text)
        document["file_type"] = file_type
        document["vectordb"] = vectordb
        print(document)
        return document
    def Text_chunker(self,text_file_path):
        with open(text_file_path, 'r', errors='ignore') as file:
            normalized_text = file.read()
        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n"]
        )
        chunks = text_splitter.split_text(normalized_text)
        return chunks

# Main processing function
def process_file(file_path):
    """Process a file (CSV, PDF, etc.) by determining its type and calling the appropriate processor."""
    file_extension = os.path.splitext(file_path)[1].lower()
    file_name = os.path.basename(file_path)
    file = os.path.splitext(file_name)
    filename = file[0]
    
    if file_extension == '.csv':
        processor = CSVProcessor(file_path)
    elif file_extension in ['.txt','.docx']:
        pythoncom.CoInitialize()
        output_path=f"{os.path.splitext(file_path)[0]}.pdf"
        convert(file_path,output_path)
        processor=PDFProcessor(output_path,file_extension)
    elif file_extension=='.pdf':
        processor = PDFProcessor(file_path,file_extension)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")
    print(filename)
    processor.process(filename)

def get_text_from_Pdf(file_path):
        doc = fitz.open(file_path)
    
    # Initialize a variable to hold the text
        full_text = ""
    
    # Iterate over each page in the document
        for page_num in range(doc.page_count):
        # Get a page
            page = doc.load_page(page_num)
        
        # Extract text from the page
            page_text = page.get_text()
        
        # Append the extracted text to the full_text string
            full_text += page_text
        
    # Close the document
        doc.close()
    
        return full_text

    # Return the processed data and vector stores for further use
   # return extracted_docs, image_embeddings, image_paths, text_vectordb # the last line is not needed
