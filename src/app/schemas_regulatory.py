"""
Request/response schemas for regulator registration (used by the API).
"""
from pydantic import BaseModel, Field, constr
from typing import List, Optional

RegId = constr(strip_whitespace=True, to_lower=True, regex=r"^[a-z0-9\-_]{2,40}$")

class DomainItem(BaseModel):
    id: RegId
    name: str = Field(min_length=2, max_length=100)

class RegisterRegulatorRequest(BaseModel):
    id: RegId                       # e.g., "qcb"
    name: str                       # "Qatar Central Bank"
    jurisdiction: str               # "Qatar"
    domains: List[DomainItem]       # checkbox selections from the UI
    # Optional seed for future extensions:
    articles: Optional[list] = None # leave None now (populated later)

class RegisterRegulatorResponse(BaseModel):
    id: str
    rulepack_path: str              # relative path to generated YAML
    config_updated: bool
