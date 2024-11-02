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

Your specific task is to determine which borough the speaker lives in based on their testimony.
The testimony may contain direct mentions of their borough, neighborhood, or community board.

Some common indicators of borough residence:
- Direct statements like "I live in Queens"
- Neighborhood mentions (e.g., "Astoria resident" implies Queens, "Williamsburg resident" implies Brooklyn)
- Community Board references (e.g., "CB3 Manhattan", "Brooklyn CB6")
- Local landmark or street references
- References to specific housing developments or projects in a borough
- Mentions of local community organizations or institutions tied to specific boroughs

If the borough cannot be determined with reasonable certainty from the testimony, mark it as "Unknown".
Do not make assumptions about borough residence based on general statements about NYC housing policy
or references to locations the speaker may only be discussing rather than residing in.

For each testimony:
1. Extract any direct mentions of boroughs or neighborhoods
2. Note any community board references
3. Identify any local landmarks or institutions mentioned
4. Evaluate the confidence level of your determination
5. Provide your final borough determination with explanation

Extract this information and provide your chain of reasoning."""

class BoroughInfo(BaseModel):
    direct_borough_mentions: List[str] = Field(
        description="List of any direct mentions of boroughs in the testimony (e.g., 'I live in Queens')",
        default_factory=list
    )
    neighborhood_references: List[str] = Field(
        description="List of any neighborhood mentions that could indicate borough (e.g., 'Astoria resident', 'Park Slope community')",
        default_factory=list
    )
    community_board_references: List[str] = Field(
        description="List of any community board references that could indicate borough",
        default_factory=list
    )
    reasoning: str = Field(
        description="Explanation of how the borough determination was made based on the evidence"
    )
    confidence_level: Literal["High", "Medium", "Low"] = Field(
        description="Confidence level in the borough determination based on available evidence"
    )
    borough: Literal[
        "Manhattan", "Queens", "Brooklyn", "Bronx", "Staten Island", "Unknown"
    ] = Field(
        description="The borough in which the individual giving testimony lives. If it is not clear in which borough the individual lives, use 'Unknown'. If you can extrapolate the borough the individual lives in from the neighborhood they stated that they live in or the community board that they are in, use that."
    )

class AnalysisResult(BaseModel):
    speaker_id: str
    testimony: str
    analysis: BoroughInfo

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
    if model_provider == "ANTHROPIC":
        client = instructor.from_anthropic(
            Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        )
    elif model_provider == "OPENAI":
        client = instructor.from_openai(OpenAI(api_key=os.getenv("OPENAI_API_KEY")))
    else:
        raise ValueError(f"Invalid model provider: {model_provider}")

    logger.info("Analyzing testimony for speaker's borough:")
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