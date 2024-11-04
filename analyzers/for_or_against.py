import logging
from typing import Dict, List, Literal
from datetime import datetime
from pydantic import BaseModel, Field
from rich.logging import RichHandler
from models import Transcript
from .analysis_utils import extract_from_testimony

# Configure logging
FORMAT = "%(message)s"
logging.basicConfig(
    level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)
logger = logging.getLogger(__name__)

# New System Prompt
SYSTEM_PROMPT = """You are analyzing testimony about the NYC City of Yes for Housing Opportunity proposal.
Your task is to determine whether each speaker is for or against the proposal based on their statements.
The City of Yes for Housing Opportunity proposal includes:
- Universal Affordability Preference (UAP)
- Residential Conversions
- Town Center Zoning
- Removing Parking Mandates
- Accessory Dwelling Units (ADUs)
- Transit-Oriented Development
- Campuses
- Small and Shared Housing
Analyze the speaker's statements carefully to determine their stance.
Extract the following information from the testimony:
- Claims Against City of Yes: List any arguments or statements that oppose the proposal.
- Claims For City of Yes: List any arguments or statements that support the proposal.
- Analysis of Testimony: Provide a brief analysis of the overall stance and key points made by the speaker.
Ensure that all extracted information is accurate and based solely on the testimony provided.
"""

class ForOrAgainstStance(BaseModel):
    claims_against_city_of_yes: List[str] = Field(
        description="A list of claims, verbatim, from the testimony that are against City of Yes."
    )
    claims_for_city_of_yes: List[str] = Field(
        description="A list of claims, verbatim, from the testimony that are in favor of City of Yes."
    )
    analysis_of_testimony_and_claims: str = Field(
        description="Your analysis of the claims for and against City of Yes, and the language of the testimony."
    )
    for_or_against: str = Field(
        description="'for' if the individual is for City of Yes, 'against' if the individual is against City of Yes."
    )

class AnalysisResult(BaseModel):
    speaker_id: str
    testimony: str
    analysis: ForOrAgainstStance

class AnalysisOutput(BaseModel):
    metadata: Dict = Field(
        default_factory=lambda: {
            "analysis_date": datetime.now().isoformat(),
            "version": "1.1"
        }
    )
    extracted_data: AnalysisResult

def extract(
    testimony_transcript: Transcript,
    model_provider: Literal["ANTHROPIC", "OPENAI"] = "ANTHROPIC",
    model_name: str = "claude-3-5-sonnet-20241022",
) -> AnalysisOutput:
    return extract_from_testimony(
        testimony_transcript=testimony_transcript,
        system_prompt=SYSTEM_PROMPT,
        output_model=AnalysisOutput,
        analysis_type="elements discussed",
        model_provider=model_provider,
        model_name=model_name,
    )