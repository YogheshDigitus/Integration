from fastapi import FastAPI
from pydantic import BaseModel
from CLIP_DocumentProcessor_Open_AI import process_file, get_text_from_Pdf
from OpenAI_rag_chain_version_testing import rag_pipeline_with_prompt, Get_summary        # Existing import 

app = FastAPI()

# Data model for the /summarize endpoint

class DocumentProcessing(BaseModel):
    Path:str

class Question(BaseModel):
    text: str

# Data model for the /respond endpoint
class RespondRequest(BaseModel):
    name: str

# Endpoint 1: Calls get_summary from CLIP_open_AI.py
@app.post("/Processing/")
async def Document_proc(data: DocumentProcessing):
    Output_dir=process_file(data.Path)
    summary=Get_summary(Output_dir)
    return {"summary": summary}

# Endpoint 1: Calls rag_pipeline_with_prompt from  OpenAI RAG pipliene
@app.post("/response/")
async def summarize_contract(data: Question):
    response = rag_pipeline_with_prompt(data.text)
    return {"response": response}

