import json
import logging
import os
from enum import Enum
from typing import List, Literal

import instructor
from anthropic import Anthropic
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from rich.logging import RichHandler

from models import Transcript

FORMAT = "%(message)s"
logging.basicConfig(
    level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)
logger = logging.getLogger(__name__)

load_dotenv()

PROMPT = """
I will provide a transcript of a testimony by an individual at a NYC City Planning Commission public hearing regarding a zoning proposal called "City of Yes for Housing Opportunity".

Your job is to determine all the elements of the proposal that the speaker discussed, whether or not they were against it.

From NYC City Planning's website, City of Yes for Housing Opportunity has the following elements:

# Universal Affordability Preference (UAP)

In recent decades, high-demand neighborhoods have lost affordable housing and become increasingly out of reach to working families.

The Universal Affordability Preference is a new tool that would allow buildings to add at least 20% more housing, if the additional homes are affordable to households earning 60% of the Area Median Income (AMI). As a result, it will deliver new affordable housing in high-cost neighborhoods across New York City to working families. 

As an example of how this policy would work, take a proposal for a 100% affordable building in a high-cost neighborhood in Manhattan like the Upper West Side.

Under the Universal Affordability Preference, the building can be built with at least 20% more space, so long as it uses that extra space for affordable housing.

The result is more affordable units in a high-cost neighborhood, and more opportunities for working families to live and thrive in New York.


# Residential Conversions

Today, outdated rules prevent underused offices and other non-residential space from converting to housing. For example, many buildings constructed after 1961, or outside the city's largest office centers, cannot be converted to housing.

City of Yes will make it easier for vacant offices and other non-residential buildings to become homes, a win-win policy to create housing, boost property values, and create more active, vibrant neighborhoods in areas that have been hard-hit by the effects of the pandemic.

Current regulations prevent buildings constructed after 1961 or outside of central business areas from converting.

This space can be put to better use as new housing.

City of Yes would legalize conversions for buildings through 1990 and expand eligibility to anywhere in the city that residential uses are allowed.

It will also allow commercial buildings to convert to a wider range of housing types, like shared housing — where apartments share kitchens and other facilities.

Letting older, outdated commercial buildings become residential is a boon to both our businesses and our workers.

City of Yes will breathe new life into our office districts and address our housing crisis.


# Town Center Zoning

New York is a city of neighborhoods, and each neighborhood is anchored by commercial corridors with shops and vibrant street life — a little town center for every community. 

Modest apartment buildings with stores on the street and apartments above exist in low-density areas across the five boroughs - most of them from the 1920s to 1950s. However, today's zoning prohibits that classic form even in areas where it's very common.

Businesses suffer from lack of customers, and people have to live farther away from goods and services.

By relegalizing housing above businesses on commercial streets in low-density areas, City of Yes will create new housing, help neighbors reach small businesses, and build vibrant mixed-use neighborhoods.


# Parking Mandates

New York City currently mandates off-street parking along with new housing even where it's not needed, driving down housing production and driving up rents. 

City of Yes would end parking mandates for new housing, as many cities across the country have successfully done. The proposal will preserve the option to add parking, but no one will be forced to build unnecessary parking.

## Issues with Current Parking Mandates

- Parking Takes Up Space
  - 2 spaces = 1 studio apartment
- Building Parking is Expensive
  - $67,500 per underground parking spot
- Parking Hinders Development
  - Especially affordable housing  
- Mandating Parking Drives Up Rent
  - For the cost of constructing 4 off-street parking spots, we could build a new home

## Benefits of Removing Parking Mandates

- Decrease Rents
  - Lifting parking mandates lowers rents
- Increase Affordable Housing Production
  - Lifting parking mandates will make it easier to build affordable housing
- Parking Is Optional
  - New housing will still include parking where it is needed, but it will no longer be required

  
# Accessory Dwelling Units (ADUs)

Across the city, small homeowners face challenges with rising costs and aging in place. Regulations limit what New Yorkers can do with their own property, which means families have to move farther away from their grandparents or grandchildren, or are forced into uncomfortably cramped houses. Meanwhile, spaces like garages go unused when improvements could make them comfortable homes.

Accessory dwelling units (backyard cottages, garage conversions) can add new homes and support homeowners without significantly changing the look and feel of a neighborhood.

For seniors fighting to stay in the neighborhood on a fixed income, or young people stretching to afford a first home, adding a small home can be life changing.

But under current rules, homeowners can't choose to use their property in this way.

City of Yes would allow "accessory dwelling units," or ADUs — like backyard cottages, garage conversions, and basement apartments — to give homeowners extra cash or provide more space for multi-generational families.

ADUs also make it easier for younger generations or caregivers to live nearby. And they can deliver big benefits while fitting in with existing buildings.


# Transit-Oriented Development

Adding housing near public transit is a commonsense approach to support convenient lifestyles, limit the need for car ownership, lower congestion, and reduce carbon emissions. Many modest apartment buildings exist in lower-density areas, most of them built between the 1920s and 1950s. 

However, current zoning bans apartment buildings like these, forcing New Yorkers into long commutes, increasing traffic congestion and worsening climate change.  

City of Yes would relegalize modest, 3- to 5-story apartment buildings where they fit best: large lots on wide streets or corners within a half-mile of public transit.
"""


class CityofYesForHousingOpportunityProposalElement(str, Enum):
    UNIVERSAL_AFFORDABILITY_PREFERENCE_UAP = "UNIVERSAL AFFORDABILITY PREFERENCE (UAP)"
    RESIDENTIAL_CONVERSIONS = "RESIDENTIAL CONVERSIONS"
    TOWN_CENTER_ZONING = "TOWN CENTER ZONING"
    REMOVING_PARKING_MANDATES = "REMOVING PARKING MANDATES"
    ACCESSORY_DWELLING_UNITS_ADU = "ACCESSORY DWELLING UNITS (ADU)"
    TRANSIT_ORIENTED_DEVELOPMENT = "TRANSIT-ORIENTED DEVELOPMENT"


class CityOfYesForHousingOpportunityElementsDiscussed(BaseModel):
    chain_of_thought: str = Field(
        description="""Think step by step about which elements of the City of Yes For Housing Opportunity proposal the individual giving testimony discussed.

        1. Which elements are explicitly stated in the testimony?
        2. Which elements are implicit in the testimony? The connection to the element must be very clear in order to be included.
        """
    )
    elements: List[CityofYesForHousingOpportunityProposalElement] = Field(
        description="A list of all the of the elements of City of Yes For Housing Opportunity discussed by the individual giving testimony. All elements discussed must be listed, whether or not the individual is for or against the proposal, and there should be no duplicates."
    )


def extract_housing_opportunity_elements_discussed(
    testimony_transcript: Transcript,
    model_provider: Literal["ANTHROPIC", "OPENAI"] = "ANTHROPIC",
    model_name: str = "claude-3-5-sonnet-20240620",
) -> CityOfYesForHousingOpportunityElementsDiscussed:
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
        "response_model": CityOfYesForHousingOpportunityElementsDiscussed,
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


def serialize_transcript(transcript: Transcript) -> str:
    text = ""
    for speaker_segment in transcript.speaker_segments:
        speaker = speaker_segment.speaker
        text += f"[NAME: {speaker.name} ROLE: {speaker.role} ORGANIZATION: {speaker.organization}]\n"
        for sentence in speaker_segment.sentences:
            text += f"{sentence.text}\n"
        text += "\n"
    return text.strip()


SPEAKERS_PATH = "speakers.json"
TRANSCRIPT_PATH = "transcript.json"
TESTIMONIES_PATH = "testimonies.json"


def main():
    transcript = Transcript.from_speakers_and_transcript_path(
        SPEAKERS_PATH, TRANSCRIPT_PATH
    )

    with open(TESTIMONIES_PATH, "r") as f:
        testimonies = json.load(f)

    testimonies_with_elements_discussed = []
    for testimony in testimonies:
        testimony_transcript = transcript.from_start_time_to_end_time(
            testimony["start_time_in_seconds"], testimony["end_time_in_seconds"]
        )
        elements_discussed = extract_housing_opportunity_elements_discussed(
            testimony_transcript
        )

        testimonies_with_elements_discussed.append(
            {
                "testimony": testimony,
                "elements_discussed": elements_discussed.model_dump(),
            }
        )

    with open("extracted-data.json", "w") as f:
        json.dump(testimonies_with_elements_discussed, f, indent=4)


if __name__ == "__main__":
    main()
