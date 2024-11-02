import logging
import os
from typing import Dict, Literal, List
from datetime import datetime
import instructor
from anthropic import Anthropic
from openai import OpenAI
from pydantic import BaseModel, Field
from rich.logging import RichHandler
from common import serialize_transcript
from models import Transcript

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

Your specific task is to determine which neighborhood the speaker lives in based on their testimony.
The testimony may contain direct mentions of their neighborhood, local landmarks, or community references.

Some common indicators of neighborhood residence:
- Direct statements like "I live in Park Slope" or "I'm a resident of Astoria"
- Community Board references that can be mapped to specific neighborhoods
- Local landmark or street references (e.g., "near Prospect Park", "off Northern Boulevard")
- References to specific housing developments or local projects
- Mentions of neighborhood-specific community organizations or institutions
- Cross-street references or specific address areas
- References to local business districts or commercial corridors

For each testimony:
1. Extract any direct mentions of neighborhoods
2. Note any specific street addresses or cross-streets
3. Document any local landmarks or institutions mentioned
4. Consider any community board references that could indicate neighborhood
5. Evaluate the confidence level of your determination
6. Provide your final neighborhood determination with detailed explanation

Be specific with neighborhood names - use commonly recognized neighborhood designations.
If the neighborhood cannot be determined with reasonable certainty, mark it as "Unknown".
Do not make assumptions based on general area references or locations the speaker may only be discussing rather than residing in.

Extract this information and provide your detailed chain of reasoning."""

class NeighborhoodInfo(BaseModel):
    direct_neighborhood_mentions: List[str] = Field(
        description="List of any direct mentions of neighborhoods in the testimony (e.g., 'I live in Park Slope')",
        default_factory=list
    )
    street_references: List[str] = Field(
        description="List of any street addresses, cross-streets, or specific location references",
        default_factory=list
    )
    local_landmarks: List[str] = Field(
        description="List of any local landmarks, institutions, or businesses mentioned that could help identify the neighborhood",
        default_factory=list
    )
    community_board_references: List[str] = Field(
        description="List of any community board references that could indicate neighborhood",
        default_factory=list
    )
    reasoning: str = Field(
        description="Detailed explanation of how the neighborhood determination was made based on the evidence"
    )
    confidence_level: Literal["High", "Medium", "Low"] = Field(
        description="Confidence level in the neighborhood determination based on available evidence"
    )
    neighborhood: str = Field(
        description="The neighborhood in which the individual giving testimony lives. If it is not clear in which neighborhood the individual lives, use 'Unknown'. Use commonly recognized neighborhood names."
    )

class AnalysisResult(BaseModel):
    speaker_id: str
    testimony: str
    analysis: NeighborhoodInfo

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
    if model_provider == "ANTHROPIC":
        client = instructor.from_anthropic(
            Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        )
    elif model_provider == "OPENAI":
        client = instructor.from_openai(OpenAI(api_key=os.getenv("OPENAI_API_KEY")))
    else:
        raise ValueError(f"Invalid model provider: {model_provider}")

    logger.info("Analyzing testimony for speaker's neighborhood:")
    logger.info(serialize_transcript(testimony_transcript))
    serialized_transcript = serialize_transcript(testimony_transcript)

    kwargs = {
        "response_model": AnalysisOutput,
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"<testimony_transcript>{serialized_transcript}</testimony_transcript>",
            },
        ],
    }
    
    if model_provider == "ANTHROPIC":
        kwargs["max_tokens"] = 8192

    response, completion = client.chat.completions.create_with_completion(**kwargs)

    logger.info(response)
    logger.info(completion)

    return response