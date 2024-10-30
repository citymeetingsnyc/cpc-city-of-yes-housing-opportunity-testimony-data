import logging
import os
from typing import List, Literal, Dict
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

def extract(
    testimony_transcript: Transcript,
    model_provider: Literal["ANTHROPIC", "OPENAI"] = "ANTHROPIC",
    model_name: str = "claude-3-5-sonnet-20241022",  # Newest anthropic model released in October
) -> AnalysisOutput:
    if model_provider == "ANTHROPIC":
        client = instructor.from_anthropic(
            Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        )
    elif model_provider == "OPENAI":
        client = instructor.from_openai(OpenAI(api_key=os.getenv("OPENAI_API_KEY")))
    else:
        raise ValueError(f"Invalid model provider: {model_provider}")

    logger.info("Analyzing testimony for stance on City of Yes proposal:")
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
        kwargs["max_tokens"] = 4096

    response, completion = client.chat.completions.create_with_completion(**kwargs)

    logger.info(response)
    logger.info(completion)

    return response
