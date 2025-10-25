"""
FastAPI app exposing the AI pipeline.
Endpoints:
  - POST /analyze  : analyze uploaded text or file
  - GET  /health   : health check
"""
from fastapi import FastAPI, UploadFile, Form
from models.inference.predict import InferencePipeline

app = FastAPI(title="AIX Hackathon Fintech Regulatory AI")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analyze")
async def analyze(
    regulator_ns: str = Form("qcb"),
    file: UploadFile = None,
    text: str = Form(None)
):
    # Load text either from uploaded file or form
    if file:
        content = (await file.read()).decode("utf-8", errors="ignore")
    else:
        content = text or ""

    pipeline = InferencePipeline(regulator_ns)
    result = pipeline.run(content)
    return {"regulator": regulator_ns, "result": result}
