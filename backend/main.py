from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from core.models import SummaryRequest, SummaryResponse
from core.nlp_engine import content_ranking, abstractive_refine, detect_src_language, translate_content
import uvicorn
import io
import PyPDF2
import docx
from docx import Document
import asyncio
from concurrent.futures import ThreadPoolExecutor

app = FastAPI(title="Lingual API", version="1.0.0")

# CORS for Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Executor for CPU tasks
executor = ThreadPoolExecutor(max_workers=3)

@app.get("/")
def health_check():
    return {"status": "ok", "system": "Multilingual Summarizer Standard"}

def extract_text_from_file(file: UploadFile) -> str:
    content = ""
    try:
        if file.filename.endswith(".pdf"):
            try:
                pdf_reader = PyPDF2.PdfReader(file.file, strict=False)
                for page in pdf_reader.pages:
                    text = page.extract_text()
                    if text:
                        content += text + "\n"
            except Exception as e:
                # Fallback or just re-raise specific error
                print(f"PyPDF2 error: {e}")
                # Sometimes file pointer needs reset if we want to try another method, 
                # but here we just fail gracefully or return what we got.
                if not content:
                     raise ValueError("Could not extract text from PDF. It might be image-based or encrypted.")

        elif file.filename.endswith(".docx"):
            doc = docx.Document(file.file)
            for paragraph in doc.paragraphs:
                content += paragraph.text + "\n"

        elif file.filename.endswith(".txt"):
            # Try utf-8 first, then fallback
            content_bytes = file.file.read()
            try:
                content = content_bytes.decode("utf-8")
            except UnicodeDecodeError:
                try:
                    content = content_bytes.decode("latin-1")
                except:
                    content = content_bytes.decode("utf-8", errors="ignore")
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Use PDF, DOCX, or TXT.")
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"File parsing error: {str(e)}")
    
    # Remove null bytes which can choke some NLP parsers
    return content.replace('\x00', '').strip()

async def run_in_threadpool(func, *args):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, func, *args)

def execute_summary_logic(text: str, sentences_count: int, language: str, mode: str, ratio: float = None):
    # 1. Detect Language
    src_lang = detect_src_language(text)
    
    # 2. Routing Logic
    summary_sents = []
    ranked_data = [] 
    metrics = {}
    final_summary_text = ""
    
    # Logic Branching
    if mode == "extractive":
        # Extractive: Summarize Source -> Translate Result
        
        summary_sents, ranked_data, metrics = content_ranking(text, top_n=sentences_count, ratio=ratio)
        extractive_text = " ".join(summary_sents)
        
        if src_lang != language and language != "auto":
            final_summary_text = translate_content(extractive_text, language)
        else:
            final_summary_text = extractive_text
            
    elif mode == "abstractive":
        # Abstractive: Translate Input -> Summarize Target
        if src_lang != language and language != "auto":
            working_text = translate_content(text, language)
        else:
            working_text = text
            
        # For heatmap, we run ranking on ORIGINAL text.
        _, ranked_data, metrics = content_ranking(text, top_n=sentences_count, ratio=ratio)
        
        # Now generate abstractive summary
        # Cap/Truncate for CPU safety (e.g. 512 tokens approx 2000 chars)
        safe_input = working_text[:3000] 
        final_summary_text = abstractive_refine(safe_input, max_length=150)
        
        summary_sents = [final_summary_text] # Placeholder for response structure

    elif mode == "hybrid":
        # Hybrid: Extractive Compression -> Refinement
        
        summary_sents, ranked_data, metrics = content_ranking(text, top_n=sentences_count, ratio=ratio)
        extractive_text = " ".join(summary_sents)
        
        if src_lang != language and language != "auto":
            translated_extractive = translate_content(extractive_text, language)
            working_text = translated_extractive
        else:
            working_text = extractive_text
            
        final_summary_text = abstractive_refine(working_text, max_length=150)

    # Fallback if empty
    if not final_summary_text:
        final_summary_text = "Could not generate summary."

    return SummaryResponse(
        original_text=text,
        summary_text=final_summary_text,
        extractive_summary=summary_sents,
        refined_summary=final_summary_text if mode != "extractive" else None,
        ranking_data=ranked_data,
        metrics=metrics
    )

@app.post("/summarize/text", response_model=SummaryResponse)
async def summarize_text_endpoint(request: SummaryRequest):
    try:
        return await asyncio.wait_for(
            run_in_threadpool(
                execute_summary_logic, 
                request.text, 
                request.sentences_count, 
                request.language, 
                request.mode,
                request.summary_ratio
            ),
            timeout=60.0
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Processing timeout (60s limit exceeded)")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize/file", response_model=SummaryResponse)
async def summarize_file_endpoint(
    file: UploadFile = File(...),
    sentences_count: int = Form(5),
    summary_ratio: float = Form(None),
    language: str = Form("en"),
    mode: str = Form("extractive")
):
    # Check size (approx 5MB limit)
    # UploadFile doesn't have size immediately available without reading or checking headers, 
    # but we can try to check content-length header if trusted, or read chunks.
    # Using spooled temp file, let's just read and check len.
    
    try:
        text = extract_text_from_file(file)
    except Exception as e:
         raise HTTPException(status_code=400, detail=f"File parsing error: {e}")

    if not text:
        raise HTTPException(status_code=400, detail="Could not extract text from file.")
        
    # Approximate 5MB text check (1 char ~ 1 byte for ascii, more for utf-8)
    if len(text) > 5 * 1024 * 1024:
         raise HTTPException(status_code=413, detail="File too large (Max 5MB text content)")

    try:
        return await asyncio.wait_for(
            run_in_threadpool(
                execute_summary_logic, 
                text, 
                sentences_count, 
                language, 
                mode,
                summary_ratio
            ),
            timeout=60.0
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Processing timeout (60s limit exceeded)")
    except Exception as e:
         raise HTTPException(status_code=500, detail=str(e))

@app.post("/export/word")
async def export_word_endpoint(request: SummaryResponse):
    # Generate Docx
    doc = Document()
    doc.add_heading('Executive Summary Report', 0)
    
    doc.add_heading('Summary', level=1)
    doc.add_paragraph(request.summary_text)
    
    doc.add_heading('Metrics', level=2)
    metrics = request.metrics
    if metrics:
        doc.add_paragraph(f"Original Length: {metrics.get('original_length', 0)} chars")
        doc.add_paragraph(f"Compression Ratio: {metrics.get('compression_ratio', 0)}")
        doc.add_paragraph(f"Sentences: {metrics.get('sentence_count', 0)}")
    
    doc.add_heading('Original Text', level=1)
    doc.add_paragraph(request.original_text)
    
    # Save to IO
    file_stream = io.BytesIO()
    doc.save(file_stream)
    file_stream.seek(0)
    
    headers = {
        'Content-Disposition': 'attachment; filename="summary_report.docx"'
    }
    return StreamingResponse(file_stream, media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document", headers=headers)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
