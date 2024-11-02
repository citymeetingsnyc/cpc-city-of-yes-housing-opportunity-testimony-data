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

SYSTEM_PROMPT = """You are analyzing testimony about the NYC City of Yes for Housing Opportunity proposal.
The City of Yes for Housing Opportunity proposal includes several key elements:
- Universal Affordability Preference (UAP)
- Residential Conversions
- Town Center Zoning
- Removing Parking Mandates
- Accessory Dwelling Units (ADUs)
- Transit-Oriented Development
- Campuses
- Small and Shared Housing

Your specific task is to identify which elements of the City of Yes for Housing Opportunity proposal the speaker discusses in their testimony.

Consider:
1. Both direct mentions and clear references to these elements
2. Discussion of impacts or concerns about specific elements
3. Suggestions or critiques related to specific elements

Note that:
- The speaker may use different terms to refer to these elements
- Some testimonies may discuss multiple elements
- Some testimonies may not discuss any specific elements
- Focus only on substantive discussion, not mere mentions in passing

Extract only the elements that are actually discussed in some detail, not just mentioned in a list or in passing."""

class ElementsInfo(BaseModel):
    elements_discussed: List[str] = Field(
        description="""
        The elements of the City of Yes for Housing Opportunity proposal that the individual discussed in their testimony.
        Possible elements include:
        - Universal Affordability Preference (UAP)
        - Residential Conversions
        - Town Center Zoning
        - Removing Parking Mandates
        - Accessory Dwelling Units (ADUs)
        - Transit-Oriented Development
        - Campuses
        - Small and Shared Housing
        If none of these elements are discussed, use an empty list.
        """
    )
    element_quotes: Dict[str, List[str]] = Field(
        description="Dictionary mapping each discussed element to relevant quotes from the testimony",
        default_factory=dict
    )
    reasoning: str = Field(
        description="Brief explanation of how the elements were identified in the testimony"
    )

class AnalysisResult(BaseModel):
    speaker_id: str
    testimony: str
    analysis: ElementsInfo

class AnalysisOutput(BaseModel):
    metadata: Dict = Field(
        default_factory=lambda: {
            "analysis_date": datetime.now().isoformat(),
            "version": "1.0"
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