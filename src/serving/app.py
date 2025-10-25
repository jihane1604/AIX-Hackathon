from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from models.inference.predict import InferencePipeline

# 1. Initialize the pipeline ONCE outside the function
try:
    # Assuming 'qcb' is a safe default for initialization
    # If the model is large, this will be the bottleneck for startup time
    model_pipeline = InferencePipeline(default_regulator_ns="qcb") 
except Exception as e:
    # Handle failure to load model at startup
    raise RuntimeError(f"Failed to initialize InferencePipeline: {e}")

app = FastAPI(title="AIX Hackathon Fintech Regulatory AI")

# ... CORS configuration 
origins = [
    # Add the origin(s) of your frontend application(s)
    "http://localhost:8000",  # Example: your local dev server
    "http://127.0.0.1:8000",
    # "https://your-production-frontend.com", 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,       # List of origins that are allowed to make requests
    allow_credentials=True,      # Allow cookies/authorization headers
    allow_methods=["*"],         # Allow all methods (POST, GET, etc.)
    allow_headers=["*"],         # Allow all headers
)

@app.post("/analyze")
async def analyze(
    regulator_ns: str = Form("qcb"),
    file: UploadFile = None,
    text: str = Form(None)
):
    # Ensure the pipeline instance uses the correct regulator namespace 
    # (assuming InferencePipeline supports updating this mid-run)
    model_pipeline.set_regulator(regulator_ns) # *Requires a new method in your InferencePipeline class*
    
    # ... rest of your logic ...
    
    # 2. Call the pre-loaded instance
    try:
        result = model_pipeline.run(content)
    except Exception as e:
        # 3. Add robust error handling
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
        
    return {"regulator": regulator_ns, "result": result}