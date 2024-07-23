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


class TalkingPointQuote(BaseModel):
    text: str = Field(
        description="A quote from the testimony that is similar to a reference talking point."
    )
    reasoning: str = Field(
        description="Your reasoning for how the quote is similar to the reference talking point."
    )


class TalkingPointsCrossReferenceAnalysis(BaseModel):
    chain_of_thought: str = Field(
        description="Think step by step about how the testimony is similar to the reference talking points."
    )
    quotes: List[TalkingPointQuote] = Field(
        description="A list of quotes from the testimony that are similar to the reference talking points. This should be empty if there are none."
    )
    analysis: str = Field(
        description="""An analysis of how closely the testimony is related to the reference talking points.

        In particular, try to discern if the testimony is:

        - HIGH_SIMILARITY_TO_TALKING_POINTS: The testimony is largely referencing similar language, quotes, claims, and arguments. It is plausible that the person may be using the reference talking points.

        - SUPPORTING_TALKING_POINTS: The testimony is similar in spirit and supporting the talking points. But the testimony is not largely referencing similar language, quotes, claims, and arguments.

        - IN_OPPOSITION: In direct opposition to reference talking points, in that there are claims and arguments that are opposed to specific points made in the reference talking points.

        - NOT_SIMILAR: Not similar to the reference talking points because it does not address any of the specific issues raised in the talking points.

        If a testimony is in opposition to the overall City of Yes for Housing Opportunity proposal, but the testimony is not directly opposing specific points made in the reference talking points, then you should classify it as "not similar".
        """
    )
    similarity: Literal[
        "HIGH_SIMILARITY_TO_TALKING_POINTS",
        "SUPPORTING_TALKING_POINTS",
        "NOT_SIMILAR",
        "IN_OPPOSITION",
    ] = Field(
        description="""The level of similarity between the testimony and the reference talking points, given your analysis.
        """
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
{% if testimony.extracted_data.quotes %}
{% for quote in testimony.extracted_data.quotes %}
> {{ quote.text }}

{{ quote.reasoning }}
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

# High Similarity To Talking Points
{{ high_similarity_to_talking_points }}

# In Opposition To Talking Points
{{ in_opposition }}

# Supporting Talking Points
{{ supporting_talking_points }}

# Not Similar To Talking Points
{{ not_similar }}
""".strip()


def generate_report(extracted_data_dir, reference_talking_points_path):
    """Generate a report from the talking points analysis."""

    high_similarity_to_talking_points = []
    supporting_talking_points = []
    not_similar = []
    in_opposition = []

    for testimony_filename in os.listdir(extracted_data_dir):
        testimony_path = os.path.join(extracted_data_dir, testimony_filename)
        with open(testimony_path, "r") as f:
            testimony = json.load(f)
            if "extracted_data" not in testimony:
                continue

        if (
            testimony["extracted_data"]["similarity"]
            == "HIGH_SIMILARITY_TO_TALKING_POINTS"
        ):
            high_similarity_to_talking_points.append(testimony)
        elif testimony["extracted_data"]["similarity"] == "SUPPORTING_TALKING_POINTS":
            supporting_talking_points.append(testimony)
        elif testimony["extracted_data"]["similarity"] == "NOT_SIMILAR":
            not_similar.append(testimony)
        elif testimony["extracted_data"]["similarity"] == "IN_OPPOSITION":
            in_opposition.append(testimony)

    with open(reference_talking_points_path, "r") as f:
        reference_talking_points = f.read()

    template = Environment().from_string(REPORT_TEMPLATE)

    overview = f"""
There were {len(high_similarity_to_talking_points)} testimonies with high alignment to the reference talking points -- largely similar language, quotes, claims, and arguments.

There were {len(in_opposition)} testimonies that were in direct opposition to the reference talking points.

There were {len(supporting_talking_points)} testimonies that were supporting the reference talking points. These were similar in spirit to the reference talking points.

There were {len(not_similar)} testimonies that were not similar to the reference talking points. These did not address any of the specific issues raised in them.
""".strip()

    table_of_contents = f"""
- [Reference Talking Points](#reference-talking-points)
- [High Similarity To Talking Points](#high-similarity-to-talking-points) ({len(high_similarity_to_talking_points)} testimonies)
- [In Opposition To Talking Points](#in-opposition-to-talking-points) ({len(in_opposition)} testimonies)
- [Supporting Talking Points](#supporting-talking-points) ({len(supporting_talking_points)} testimonies)
- [Not Similar To Talking Points](#not-similar-to-talking-points) ({len(not_similar)} testimonies)
""".strip()

    rendered_report = template.render(
        reference_talking_points=reference_talking_points,
        high_similarity_to_talking_points=render_testimonies(
            high_similarity_to_talking_points
        ),
        supporting_talking_points=render_testimonies(supporting_talking_points),
        not_similar=render_testimonies(not_similar),
        in_opposition=render_testimonies(in_opposition),
        table_of_contents=table_of_contents,
        overview=overview,
    )

    return rendered_report


def render_testimonies(testimonies):
    template = Environment().from_string(TESTIMONIES_TEMPLATE)
    return template.render(testimonies=testimonies)
