from fastapi import APIRouter, UploadFile, HTTPException
from services.embedding_service import process_and_store_pdf, langgraph_agent
from models.pdf_embedding import QueryRequest
router = APIRouter()
from typing import List, Dict

@router.post("/upload-pdf/")
async def upload_pdf(file: UploadFile):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    embeddings = await process_and_store_pdf(file)
    return {"message": "PDF processed successfully", "embeddings": embeddings}


@router.post("/query/")
async def query_database(query: str):
    try:
        response = await langgraph_agent(query)
        return {"message": "Query processed successfully", "response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")
