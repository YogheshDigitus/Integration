from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
from pydantic import BaseModel
from CLIP_DocumentProcessor_Open_AI_with_image import process_file, get_text_from_Pdf
from OpenAI_rag_chain_version_testing import rag_pipeline_with_prompt, Get_summary        # Existing import 
from QWEN2_5_1_5B_ragechain_test import rag_pipeline_with_prompt as Local_rag 
app = FastAPI()

# Set the origins you want to allow, or use ["*"] to allow all (for development only)
origins = [
    "http://localhost:3000",        # If frontend is running locally
    "http://127.0.0.1:3000",
    "http://your-frontend-domain.com",  # Replace with your actual frontend domain
    "*"  # Allow all origins (not recommended in production)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Or ["*"] for public access
    allow_credentials=True,
    allow_methods=["*"],     # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],     # Allow all headers
)
# Data model for the /summarize endpoint

class Question(BaseModel):
    text: str

# Data model for the /respond endpoint
class RespondRequest(BaseModel):
    name: str

# Endpoint 1: Calls get_summary from CLIP_open_AI.py
@app.post("/Processing/")
async def Document_proc(file: UploadFile = File(...)):
    file_location = f"./uploaded_files/{file.filename}"
    # Save the file directly without reading it
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    Output_dir=process_file(file_location)
    print("The output directory is \n")
    print(Output_dir)
    context=get_text_from_Pdf(Output_dir)
    summary=Get_summary(context)
    return {"summary": summary}

# Endpoint 1: Calls rag_pipeline_with_prompt from  OpenAI RAG pipliene
@app.post("/response/")
async def summarize_contract(data: Question):
    response,image_path = rag_pipeline_with_prompt(data.text)
    return {"response": response,"image":image_path}

# Endpoint 1: Calls get_summary from CLIP_open_AI.py
#@app.post("/Local_model_Processing/")
#async def Document_proc(file: UploadFile = File(...)):
#    file_location = f"./uploaded_files/{file.filename}"
    # Save the file directly without reading it
#    with open(file_location, "wb") as buffer:
#        shutil.copyfileobj(file.file, buffer)
#    Output_dir=process_file(file_location)
#    print(Output_dir)
#    context=get_text_from_Pdf(Output_dir)
#    summary=Lcoal_summary(context)
#    return {"summary": summary}

# Endpoint 1: Calls rag_pipeline_with_prompt from  OpenAI RAG pipliene
@app.post("/Local_model_response/")
async def summarize_contract(data: Question):
    response,image_path = Local_rag(data.text)
    return {"response": response,"image":image_path}
