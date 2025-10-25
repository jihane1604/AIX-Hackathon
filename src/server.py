"""
FastAPI service exposing a POST endpoint to register a regulator.
Your web UI posts JSON (id, name, jurisdiction, domains[]) after user uploads their corpus.
This generates the rule-pack YAML and updates config/config.yaml automatically.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas_regulator import RegisterRegulatorRequest, RegisterRegulatorResponse
from app.registry import ensure_rulepack, ensure_config_entry

app = FastAPI(title="AIX-HACKATHON – Regulator Registry API")

# Relaxed CORS for hackathon testing; tighten in prod.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

@app.post("/regulators", response_model=RegisterRegulatorResponse)
def register_regulator(payload: RegisterRegulatorRequest):
    try:
        rulepack_rel = ensure_rulepack(payload)
        updated = ensure_config_entry(payload, rulepack_rel)
        return RegisterRegulatorResponse(
            id=payload.id,
            rulepack_path=rulepack_rel,
            config_updated=updated
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
