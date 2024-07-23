import logging
import os
from enum import Enum
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

PROMPT = """
I will provide a transcript of a testimony by an individual at a NYC City Planning Commission public hearing regarding a zoning proposal called "City of Yes for Housing Opportunity".

Your job is to determine all the elements of the proposal that the speaker discussed.

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

# Campuses

Across the city, many residential, faith-based, or other campuses have underused space that they could turn into housing. That new construction can pay for repairs to existing buildings, breathe new life into community institutions, and help address our housing crisis.

Today, arbitrary rules get in the way. 

For example, if existing buildings are too tall or too far back from the street, zoning prohibits new development on the property - even if the new developments would comply with current regulations.

City of Yes would make it easier for campuses to add new buildings if they wish to by removing obstacles and streamlining outdated rules. The new buildings could bring money for repairs, new facilities, and housing.

# Small and Shared Housing

NYC banned shared housing in the 1950s and apartment buildings full of studio apartments in the 1960s. This has contributed to the homelessness crisis in the decades since, and forced people who would prefer to live alone into living with roommates.

City of Yes for Housing Opportunity would re-legalize housing with shared kitchens or other common facilities. It would also allow buildings with more studios and one-bedrooms for the many New Yorkers who want to live alone but don't have that option today.

These apartments are important for so many people - recent college graduates, older households that are downsizing, and everyone who lives with roommates but would prefer to live alone. Allowing more small and shared apartments will also open up larger, family-sized apartments otherwise be occupied by roommates.
"""


class CityofYesForHousingOpportunityProposalElement(str, Enum):
    UNIVERSAL_AFFORDABILITY_PREFERENCE_UAP = "UNIVERSAL AFFORDABILITY PREFERENCE (UAP)"
    RESIDENTIAL_CONVERSIONS = "RESIDENTIAL CONVERSIONS"
    TOWN_CENTER_ZONING = "TOWN CENTER ZONING"
    REMOVING_PARKING_MANDATES = "REMOVING PARKING MANDATES"
    ACCESSORY_DWELLING_UNITS_ADU = "ACCESSORY DWELLING UNITS (ADU)"
    TRANSIT_ORIENTED_DEVELOPMENT = "TRANSIT-ORIENTED DEVELOPMENT"
    CAMPUSES = "CAMPUSES"
    SMALL_AND_SHARED_HOUSING = "SMALL AND SHARED HOUSING"


class Quote(BaseModel):
    text: str = Field(
        description="The quote from the testimony that indicates the speaker is discussing an element of the proposal. It is irrelevant whether the speaker is for or against the proposal."
    )
    reasoning: str = Field(
        description="Your reasoning for why the quote indicates that the speaker is discussing an element of the proposal. It should tie back directly to the proposal put forth by City Planning. It is irrelevant whether the speaker is for or against the proposal."
    )


class CityOfYesForHousingOpportunityElementDiscussionAnalysis(BaseModel):
    element_of_proposal: Literal[
        "UNIVERSAL_AFFORDABILITY_PREFERENCE_UAP",
        "RESIDENTIAL_CONVERSIONS",
        "TOWN_CENTER_ZONING",
        "REMOVING_PARKING_MANDATES",
        "ACCESSORY_DWELLING_UNITS_ADU",
        "TRANSIT_ORIENTED_DEVELOPMENT",
        "CAMPUSES",
        "SMALL_AND_SHARED_HOUSING",
    ] = Field(
        description="The element of the City of Yes For Housing Opportunity proposal that you are looking for in the testimony."
    )
    quotes_indicating_discussion: List[Quote] = Field(
        description="A list of quotes from the testimony that indicate the speaker is discussing the element of the proposal you are analyzing. This should be empty if the element is not discussed."
    )
    element_is_discussed: bool = Field(
        description="Whether or not the element of the proposal is discussed by the testimony."
    )


class UAPAnalysis(CityOfYesForHousingOpportunityElementDiscussionAnalysis):
    element_of_proposal: Literal["UNIVERSAL_AFFORDABILITY_PREFERENCE_UAP"] = Field(
        description="The element of the City of Yes For Housing Opportunity proposal that you are looking for in the testimony."
    )


class ResidentialConversionsAnalysis(
    CityOfYesForHousingOpportunityElementDiscussionAnalysis
):
    element_of_proposal: Literal["RESIDENTIAL_CONVERSIONS"] = Field(
        description="The element of the City of Yes For Housing Opportunity proposal that you are looking for in the testimony."
    )


class TownCenterZoningAnalysis(CityOfYesForHousingOpportunityElementDiscussionAnalysis):
    element_of_proposal: Literal["TOWN_CENTER_ZONING"] = Field(
        description="The element of the City of Yes For Housing Opportunity proposal that you are looking for in the testimony."
    )


class RemovingParkingMandatesAnalysis(
    CityOfYesForHousingOpportunityElementDiscussionAnalysis
):
    element_of_proposal: Literal["REMOVING_PARKING_MANDATES"] = Field(
        description="The element of the City of Yes For Housing Opportunity proposal that you are looking for in the testimony."
    )


class AccessoryDwellingUnitsAnalysis(
    CityOfYesForHousingOpportunityElementDiscussionAnalysis
):
    element_of_proposal: Literal["ACCESSORY_DWELLING_UNITS_ADU"] = Field(
        description="The element of the City of Yes For Housing Opportunity proposal that you are looking for in the testimony."
    )


class TransitOrientedDevelopmentAnalysis(
    CityOfYesForHousingOpportunityElementDiscussionAnalysis
):
    element_of_proposal: Literal["TRANSIT_ORIENTED_DEVELOPMENT"] = Field(
        description="The element of the City of Yes For Housing Opportunity proposal that you are looking for in the testimony."
    )


class CampusesAnalysis(CityOfYesForHousingOpportunityElementDiscussionAnalysis):
    element_of_proposal: Literal["CAMPUSES"] = Field(
        description="The element of the City of Yes For Housing Opportunity proposal that you are looking for in the testimony."
    )


class SmallAndSharedHousingAnalysis(
    CityOfYesForHousingOpportunityElementDiscussionAnalysis
):
    element_of_proposal: Literal["SMALL_AND_SHARED_HOUSING"] = Field(
        description="The element of the City of Yes For Housing Opportunity proposal that you are looking for in the testimony."
    )


class CityOfYesForHousingOpportunityAnalysis(BaseModel):
    uap_analysis: UAPAnalysis = Field(
        description="An analysis of whether or not the UAP element of the City of Yes For Housing Opportunity proposal is discussed in the testimony."
    )
    residential_conversions_analysis: ResidentialConversionsAnalysis = Field(
        description="An analysis of whether or not the Residential Conversions element of the City of Yes For Housing Opportunity proposal is discussed in the testimony."
    )
    town_center_zoning_analysis: TownCenterZoningAnalysis = Field(
        description="An analysis of whether or not the Town Center Zoning element of the City of Yes For Housing Opportunity proposal is discussed in the testimony."
    )
    removing_parking_mandates_analysis: RemovingParkingMandatesAnalysis = Field(
        description="An analysis of whether or not the Removing Parking Mandates element of the City of Yes For Housing Opportunity proposal is discussed in the testimony."
    )
    accessory_dwelling_units_analysis: AccessoryDwellingUnitsAnalysis = Field(
        description="An analysis of whether or not the Accessory Dwelling Units element of the City of Yes For Housing Opportunity proposal is discussed in the testimony."
    )
    transit_oriented_development_analysis: TransitOrientedDevelopmentAnalysis = Field(
        description="An analysis of whether or not the Transit-Oriented Development element of the City of Yes For Housing Opportunity proposal is discussed in the testimony."
    )
    campuses_analysis: CampusesAnalysis = Field(
        description="An analysis of whether or not the Campuses element of the City of Yes For Housing Opportunity proposal is discussed in the testimony."
    )
    small_and_shared_housing_analysis: SmallAndSharedHousingAnalysis = Field(
        description="An analysis of whether or not the Small and Shared Housing element of the City of Yes For Housing Opportunity proposal is discussed in the testimony."
    )


def extract(
    testimony_transcript: Transcript,
    model_provider: Literal["ANTHROPIC", "OPENAI"] = "ANTHROPIC",
    model_name: str = "claude-3-5-sonnet-20240620",
) -> CityOfYesForHousingOpportunityAnalysis:
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
        "response_model": CityOfYesForHousingOpportunityAnalysis,
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
