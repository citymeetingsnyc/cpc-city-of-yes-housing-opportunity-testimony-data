import json
import logging
import os
from typing import List, Literal

import instructor
from anthropic import Anthropic
from jinja2 import Environment
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

PROMPT = """
I will provide a transcript of a testimony by an individual at a NYC City Planning Commission public hearing regarding a zoning proposal called "City of Yes for Housing Opportunity".

I will also provide a list of reference talking points from an organization.

Your job is to cross reference the testimony with the reference talking points and determine how closely the testimony is related to the reference talking points.
"""


class TalkingPointEvidence(BaseModel):
    """Evidence of a talking point being referenced in the testimony."""

    quote: str = Field(
        description="A quote from the testimony that relates directly to a reference talking point. It must support at least one reference talking point directly."
    )
    analysis: str = Field(
        description="""Your analysis of how closely the quote relates to the reference talking point. You must classify the quote as one of the following:

        - IDENTICAL_POINT: The quote makes an identical claim or argument to a talking point. The language does not need to be exactly the same, but the claim or argument must be the same to classify it as "IDENTICAL_POINT".
        - SUPPORTS_POINT: The quote supports the reference talking point, even though the claim or argument is not the same.
        - TENUOUS_LINK: There may be a link between the quote and the reference talking point, but it is tenuous at best.
        """
    )
    closeness_to_reference_talking_point: Literal[
        "IDENTICAL_POINT", "SUPPORTS_POINT", "TENUOUS_LINK"
    ] = Field(
        description="""The closeness between the quote and the reference talking point, based on the analysis."""
    )


class TalkingPointsCrossReferenceAnalysis(BaseModel):
    chain_of_thought: str = Field(
        description="Think step by step about how the testimony is similar to the reference talking points."
    )
    talking_point_evidence: List[TalkingPointEvidence] = Field(
        description="A list of quotes from the testimony that are similar to the reference talking points. This should be empty if there are none."
    )
    analysis: str = Field(
        description="""An analysis of how closely the testimony is related to the reference talking points.

        You must classify the testimony as one of the following:

        - HIGH_ALIGNMENT: The testimony is very closely related to the reference talking points, often making the exact same claims and arguments. The language may be similar and it is plausible that the person is using the reference talking points in their testimony.
        - SUPPORTS_TALKING_POINTS: The testimony is supporting the same claims and arguments as the talking points, but it does not make the same ones or use very similar language.
        - NOT_ALIGNED: The testimony is not closely related to the reference talking points.
        """
    )
    similarity: Literal[
        "HIGH_ALIGNMENT",
        "SUPPORTS_TALKING_POINTS",
        "NOT_ALIGNED",
    ] = Field(
        description="""The level of similarity between the testimony and the reference talking points, given your analysis."""
    )


def extract(
    testimony_transcript: Transcript,
    reference_talking_points: str,
    model_provider: Literal["ANTHROPIC", "OPENAI"] = "ANTHROPIC",
    model_name: str = "claude-3-5-sonnet-20240620",
) -> TalkingPointsCrossReferenceAnalysis:
    if model_provider == "ANTHROPIC":
        client = instructor.from_anthropic(
            Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        )
    elif model_provider == "OPENAI":
        client = instructor.from_openai(OpenAI(api_key=os.getenv("OPENAI_API_KEY")))
    else:
        raise ValueError(f"Invalid model provider: {model_provider}")

    logger.info("Extracting elements discussed from testimony:")
    logger.info(serialize_transcript(testimony_transcript))

    serialized_transcript = serialize_transcript(testimony_transcript)

    kwargs = {
        "response_model": TalkingPointsCrossReferenceAnalysis,
        "model": model_name,
        "messages": [
            {"role": "system", "content": PROMPT},
            {
                "role": "user",
                "content": f"<testimony_transcript>{serialized_transcript}</testimony_transcript>\n\n<reference_talking_points>{reference_talking_points}</reference_talking_points>",
            },
        ],
    }

    if model_provider == "ANTHROPIC":
        kwargs["max_tokens"] = 4096

    response, completion = client.chat.completions.create_with_completion(**kwargs)

    logger.info(response)
    logger.info(completion)

    return response


TESTIMONIES_TEMPLATE = """
{% for testimony in testimonies %}
## {{ testimony.testimony.name }}

[{{ testimony.testimony.citymeetings_url }}]({{ testimony.testimony.   citymeetings_url }})

- **Similarity**: {{ testimony.extracted_data.similarity }}
- **For/Against COYHO:** {{ testimony.testimony.for_or_against }}
- **Stated Affiliations:** {{ testimony.testimony.stated_affiliations|join(", ") }}
- **Borough:** {% if testimony.testimony.borough != "Unknown" %}{{ testimony.testimony.borough }}{% else %}-{% endif %}
- **Neighborhood:** {% if testimony.testimony.neighborhood != "Unknown" %}{{ testimony.testimony.neighborhood }}{% else %}-{% endif %}
### Quotes
{% if testimony.extracted_data.talking_point_evidence %}
{% for talking_point_evidence in testimony.extracted_data.talking_point_evidence %}
> {{ talking_point_evidence.quote }}

- **Closeness to Reference Talking Point:** {{ talking_point_evidence.closeness_to_reference_talking_point }}
- **Analysis:** {{ talking_point_evidence.analysis }}
{% endfor %}
{% else %}
No quotes found.
{% endif %}
### Analysis
{{ testimony.extracted_data.analysis }}
{% endfor %}
""".strip()

REPORT_TEMPLATE = """
{{ overview }}

# Table of Contents
{{ table_of_contents }}

# Reference Talking Points
```
{{ reference_talking_points }}
```

# High Alignment
{{ high_alignment }}

# Supports Talking Points
{{ supports_talking_points }}

# Not Aligned
{{ not_aligned }}
""".strip()


def generate_report(extracted_data_dir, reference_talking_points_path):
    """Generate a report from the talking points analysis."""

    high_alignment = []
    supports_talking_points = []
    not_aligned = []

    for testimony_filename in os.listdir(extracted_data_dir):
        testimony_path = os.path.join(extracted_data_dir, testimony_filename)
        with open(testimony_path, "r") as f:
            testimony = json.load(f)
            if "extracted_data" not in testimony:
                continue

        if testimony["extracted_data"]["similarity"] == "HIGH_ALIGNMENT":
            high_alignment.append(testimony)
        elif testimony["extracted_data"]["similarity"] == "SUPPORTS_TALKING_POINTS":
            supports_talking_points.append(testimony)
        elif testimony["extracted_data"]["similarity"] == "NOT_ALIGNED":
            not_aligned.append(testimony)

    with open(reference_talking_points_path, "r") as f:
        reference_talking_points = f.read()

    template = Environment().from_string(REPORT_TEMPLATE)

    overview = f"""
There were {len(high_alignment)} testimonies with high alignment to the reference talking points -- largely similar language, quotes, claims, and arguments.

There were {len(supports_talking_points)} testimonies that were supporting the reference talking points. These were similar in spirit to the reference talking points.

There were {len(not_aligned)} testimonies that were not aligned to the reference talking points. These did not address any of the specific issues raised in them.
""".strip()

    table_of_contents = f"""
- [Reference Talking Points](#reference-talking-points)
- [High Alignment](#high-alignment) ({len(high_alignment)} testimonies)
- [Supports Talking Points](#supports-talking-points) ({len(supports_talking_points)} testimonies)
- [Not Aligned](#not-aligned) ({len(not_aligned)} testimonies)
""".strip()

    rendered_report = template.render(
        reference_talking_points=reference_talking_points,
        high_alignment=render_testimonies(high_alignment),
        supports_talking_points=render_testimonies(supports_talking_points),
        not_aligned=render_testimonies(not_aligned),
        table_of_contents=table_of_contents,
        overview=overview,
    )

    return rendered_report


def render_testimonies(testimonies):
    template = Environment().from_string(TESTIMONIES_TEMPLATE)
    return template.render(testimonies=testimonies)
