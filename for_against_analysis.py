import logging
import os
from typing import List, Literal

import instructor
from anthropic import Anthropic
from openai import OpenAI
from pydantic import BaseModel, Field
from rich.logging import RichHandler

from common import serialize_transcript
from models import Transcript

FORMAT = "%(message)s"
logging.basicConfig(
    level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)
logger = logging.getLogger(__name__)

CITY_OF_YES_INFO = """
City of Yes for Housing Opportunity has the following elements:

1. Universal Affordability Preference (UAP)
2. Residential Conversions
3. Town Center Zoning
4. Removing Parking Mandates
5. Accessory Dwelling Units (ADUs)
6. Transit-Oriented Development
7. Campuses
8. Small and Shared Housing

These elements aim to address various housing issues in New York City, including affordability, underutilized spaces, and zoning restrictions.
"""

PROMPT = f"""
You are an AI assistant tasked with analyzing testimonies from a NYC City Planning Commission public hearing regarding the "City of Yes for Housing Opportunity" zoning proposal.

Your job is to determine whether the speaker is for or against the proposal based on their testimony.

Here's a brief overview of the City of Yes for Housing Opportunity proposal:

{CITY_OF_YES_INFO}

Please analyze the testimony and determine if the speaker is for or against the proposal. Provide reasoning and relevant quotes to support your conclusion.
"""


class Quote(BaseModel):
    text: str = Field(description="A relevant quote from the testimony")
    reasoning: str = Field(
        description="Explanation of how this quote supports the for/against stance"
    )


class ForAgainstAnalysis(BaseModel):
    stance: Literal["FOR", "AGAINST"] = Field(
        description="Whether the speaker is for or against the City of Yes proposal"
    )
    confidence: float = Field(
        description="Confidence level in the stance determination (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
    )
    reasoning: str = Field(description="Overall reasoning for the stance determination")
    supporting_quotes: List[Quote] = Field(
        description="Quotes from the testimony that support the stance determination"
    )


def extract(
    testimony_transcript: Transcript,
    model_provider: Literal["ANTHROPIC", "OPENAI"] = "ANTHROPIC",
    model_name: str = "claude-3-5-sonnet-20240620",
) -> ForAgainstAnalysis:
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
        "response_model": ForAgainstAnalysis,
        "model": model_name,
        "messages": [
            {"role": "system", "content": PROMPT},
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
