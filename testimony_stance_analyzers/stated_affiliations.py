import logging
import os
from typing import Dict, List, Literal
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

Your specific task is to identify any stated affiliations of the speaker based on their testimony.

An affiliation is defined as:
- Organizations or associations they explicitly state they are a member of or represent
- Community boards or government bodies they officially serve on
- Professional organizations or unions they belong to
- Advocacy groups or coalitions they are part of
- Business or industry associations they represent

What is NOT considered an affiliation:
- Where they live (e.g., "resident of Brooklyn")
- Their job title or role without an organization
- General demographic information
- Places they've worked in the past but don't currently represent
- Organizations they merely mention but don't claim membership in

For each testimony:
1. Identify explicit statements of affiliation (e.g., "I represent...", "I am a member of...", "Speaking on behalf of...")
2. Note any official positions mentioned with organizations
3. Distinguish between current active affiliations versus past or passive connections
4. List only clearly stated affiliations, not implied ones

If you're unsure whether something counts as an affiliation, explain your reasoning.
Focus on extracting only current, active organizational affiliations that the speaker explicitly states."""

class AffiliationInfo(BaseModel):
    stated_affiliations: List[str] = Field(
        description="""
        The affiliations that the individual giving testimony has stated that they have.

        An affiliation can include an organization/association or government body they are a part of.

        An affiliation does not include where the individual lives, or their role.

        Affiliations are purely entities that the individual states they are a part of.

        This value should be a list of these affiliations.
        """
    )
    reasoning: str = Field(
        description="Explanation of how the affiliations were determined, including any significant mentions that were excluded and why"
    )
    confidence_level: Literal["High", "Medium", "Low"] = Field(
        description="Confidence level in the affiliation determinations based on how explicitly they were stated"
    )
    raw_affiliation_statements: List[str] = Field(
        description="Direct quotes from the testimony that indicate affiliations",
        default_factory=list
    )

class AnalysisResult(BaseModel):
    speaker_id: str
    testimony: str
    analysis: AffiliationInfo

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

    logger.info("Analyzing testimony for speaker's stated affiliations:")
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