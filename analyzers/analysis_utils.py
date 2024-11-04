import logging
import os
from typing import Type, Literal
from pydantic import BaseModel
import instructor
from anthropic import Anthropic
from openai import OpenAI
from rich.logging import RichHandler
from models import Transcript
from common import serialize_transcript

# This is a helper function that is only used by the following analysis modules in this folder:
# - Borough analysis
# - Neighborhood analysis
# - Stated affiliations
# - Elements discussed
# - For or against

# Example usage syntax:
# The following command demonstrates how to run the main analysis file (analyze.py) with specific modules and data:
# poetry run python analyze.py {borough|neighborhood|affiliations|elements|for-against} city-council-data

# - Replace the placeholder in braces with one of the module names (e.g., 'borough' or 'neighborhood').
# - "city-council-data" refers to the input folder dataset for analysis.

# Configure logging
FORMAT = "%(message)s"
logging.basicConfig(
    level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)
logger = logging.getLogger(__name__)

def extract_from_testimony(
    testimony_transcript: Transcript,
    system_prompt: str,
    output_model: Type[BaseModel],
    analysis_type: str = "testimony",
    model_provider: Literal["ANTHROPIC", "OPENAI"] = "ANTHROPIC",
    model_name: str = "claude-3-5-sonnet-20241022",
) -> BaseModel:
    """
    Generic function to analyze testimony using LLM.
    
    Args:
        testimony_transcript: The transcript to analyze
        system_prompt: The system prompt to use for analysis
        output_model: The Pydantic model class for the output
        analysis_type: Description of what's being analyzed (for logging)
        model_provider: Which AI provider to use
        model_name: Which model to use
        
    Returns:
        Instance of output_model containing analysis results
    """
    if model_provider == "ANTHROPIC":
        client = instructor.from_anthropic(
            Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        )
    elif model_provider == "OPENAI":
        client = instructor.from_openai(OpenAI(api_key=os.getenv("OPENAI_API_KEY")))
    else:
        raise ValueError(f"Invalid model provider: {model_provider}")

    logger.info(f"Analyzing testimony for {analysis_type}:")
    logger.info(serialize_transcript(testimony_transcript))
    serialized_transcript = serialize_transcript(testimony_transcript)

    kwargs = {
        "response_model": output_model,
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
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