import logging
import os
from typing import List

import instructor
from anthropic import Anthropic
from pydantic import BaseModel, Field

from core.models import CPCCityOfYesTestimony, TranscriptChapter
from core.schemas import Transcript, TranscriptSpeakerSegment

from . import ApplicationCommand

logger = logging.getLogger(__name__)


CITY_OF_YES_SUMMARY = """
    # City of Yes for Housing Opportunity Overview

    Every day, New Yorkers struggle with our city's housing shortage. High housing costs, long commutes, cramped apartments, and instability are all the result of a lack of options.

    Outdated, restrictive, and complicated zoning laws limit opportunities to create new homes and make those that do get built more expensive.

    City of Yes for Housing Opportunity is a zoning reform proposal that would address the housing crisis by making it possible to build a little more housing in every neighborhood.

    The proposal will enter public review in spring of 2024, receiving input from community boards and borough presidents before going to the City Planning Commission for a vote. If passed, it is anticipated to come for a vote before the City Council by the end of the calendar year.

    # What would City of Yes for Housing Opportunity do?

    ## Universal Affordability Preference (UAP)

    In recent decades, high-demand neighborhoods have lost affordable housing and become increasingly out of reach to working families.

    The Universal Affordability Preference is a new tool that would allow buildings to add at least 20% more housing, if the additional homes are affordable to households earning 60% of the Area Median Income (AMI). As a result, it will deliver new affordable housing in high-cost neighborhoods across New York City to working families.

    ## Residential Conversions

    Today, outdated rules prevent underused offices and other non-residential space from converting to housing. For example, many buildings constructed after 1961, or outside the city's largest office centers, cannot be converted to housing.

    City of Yes will make it easier for vacant offices and other non-residential buildings to become homes, a win-win policy to create housing, boost property values, and create more active, vibrant neighborhoods in areas that have been hard-hit by the effects of the pandemic.

    ## Town Center Zoning

    New York is a city of neighborhoods, and each neighborhood is anchored by commercial corridors with shops and vibrant street life - a little town center for every community.

    Modest apartment buildings with stores on the street and apartments above exist in low-density areas across the five boroughs - most of them from the 1920s to 1950s. Today, zoning prohibits that classic form even in areas where it's very common.

    By relegalizing housing above businesses on commercial streets in low-density areas, City of Yes will create new housing, help neighbors reach small businesses, and build vibrant mixed-use neighborhoods.

    ## Removing Parking Mandates

    New York City currently mandates off-street parking along with new housing even where it's not needed, driving down housing production and driving up rents.

    City of Yes would end parking mandates for new housing, as many cities across the country have successfully done. The proposal will preserve the option to add parking, but no one will be forced to build unnecessary parking.

    ## Accessory Dwelling Units (ADUs)

    Across the city, homeowners face challenges with rising costs, aging in place and accommodating their families. Regulations limit what New Yorkers can do with their property, which means families have to move farther away from their grandparents or grandchildren, or are forced into uncomfortably cramped houses. Meanwhile, spaces like garages go unused when improvements could make them comfortable homes.

    For seniors fighting to stay in the neighborhood on a fixed income, or young people stretching to afford a first home, adding a small rental unit can be life changing. But under current rules, New York City homeowners can't choose to use their property in this way.

    City of Yes would allow "accessory dwelling units," or ADUs - which include backyard cottages, garage conversions, and basement apartments. Cities across the country have already legalized accessory dwelling units because they support homeowners and provide more space for multi-generational families without significantly changing the look and feel of a neighborhood.

    Accessory dwelling units also make it easier for younger generations or caregivers to live nearby. And they can deliver big benefits while fitting in with existing buildings.

    ## Transit Oriented Development

    Adding housing near public transit is a commonsense approach to support convenient lifestyles, limit the need for car ownership, lower congestion, and reduce carbon emissions. Many modest apartment buildings exist in lower-density areas , most of them built between the 1920s and 1950s.

    However, current zoning bans apartment buildings like these, forcing New Yorkers into long commutes, increasing traffic congestion and worsening climate change.

    City of Yes would relegalize modest, 3- to 5-story apartment buildings where they fit best: large lots on wide streets or corners within a half-mile of public transit.

    ## Campuses

    Across the city, many residential, faith-based, or other campuses have underused space that they could turn into housing. That new construction can pay for repairs to existing buildings, breathe new life into community institutions, and help address our housing crisis.

    Today, arbitrary rules get in the way. If existing buildings are too tall or too far back from the street, for instance, zoning prohibits new development on the property -- even if the new developments would comply with current regulations.By removing obstacles and streamlining outdated rules, City of Yes would make it easier for campuses to add new buildings if they wish to. The new buildings could bring money for repairs, new facilities, and housing.

    City of Yes advances key recommendations from the Where We Live NYC Plan. Where We Live NYC is the outcome of an in-depth two-year process with over 150 community partners that identified strategies for fair housing and equity.

    ## Small and Shared Housing

    NYC banned shared housing in the 1950s and apartment buildings full of studio apartments in the 1960s. This has contributed to the homelessness crisis in the decades since, and forced people who would prefer to live alone into living with roommates.

    City of Yes for Housing Opportunity would re-legalize housing with shared kitchens or other common facilities. It would also allow buildings with more studios and one-bedrooms for the many New Yorkers who want to live alone but don't have that option today.

    These apartments are important for so many people - recent college graduates, older households that are downsizing, and everyone who lives with roommates but would prefer to live alone. Allowing more small and shared apartments will also open up larger, family-sized apartments otherwise be occupied by roommates.

    ---

    The zoning text changes are long and complex, but the above is the summary and intent from the NYC Department of City Planning.
    """

SYSTEM_PROMPT = """
I will provide you a portion of a transcript of a NYC City Planning Commission public hearing on an NYC initiative called "City of Yes for Housing Opportunity"

The portion of the transcript you are provided is public testimony by one individual.

Your job is to extract a set of properties about the testimony.

For context, here is information about the initiative directly from the NYC Department of City Planning website:

---

{CITY_OF_YES_SUMMARY}
"""


class TestimonyProperties(BaseModel):
    """Properties extracted from this individual's testimony at the City of Yes for Housing Opportunity Hearing."""

    # for_or_against: Literal["For", "Against"] = Field(
    #     description="Whether the individual giving testimony is for or against the City of Yes for Housing Opportunity."
    # )
    # borough: Literal[
    #     "Manhattan", "Queens", "Brooklyn", "Bronx", "Staten Island", "Unknown"
    # ] = Field(
    #     description="The borough in which the individual giving testimony lives. If it is not clear in which borough the individual lives, use 'Unknown'. If you can extrapolate the borough the individual lives in from the neighborhood they stated that they live in or the community board that they are in, use that."
    # )
    # neighborhood: str = Field(
    #     description="The neighborhood in which the individual giving testimony lives. If it is not clear in which neighborhood the individual lives, use 'Unknown'."
    # )
    # stated_affiliations: List[str] = Field(
    #     description="""
    #     The affiliations that the individual giving testimony has stated that they have.

    #     An affiliation can include an organization/association or government body they are a part of.

    #     An affiliation does not include where the individual lives, or their role.

    #     Affiliations are purely entities that the individual states they are a part of.

    #     This value should be a list of these affiliations.
    #     """
    # )
    primary_talking_points: List[str] = Field(
        description="The primary talking points of the individual giving testimony. These are the main points they are making in their testimony."
    )


class Command(ApplicationCommand):
    help = ""

    def add_arguments(self, parser):
        parser.add_argument(
            "--remarks-only",
            action="store_true",
            help="Only extract data from dan garodnik's opening and closing remarks.",
        )

    def run(
        self,
        remarks_only,
        *args,
        **options,
    ):
        if remarks_only:
            self.__extract_data(TranscriptChapter.objects.get(id=9119))
            self.__extract_data(TranscriptChapter.objects.get(id=9120))
            self.__extract_data(TranscriptChapter.objects.get(id=9529))
            return

        cpc_city_of_yes_testimonies = CPCCityOfYesTestimony.objects.all()

        total = len(cpc_city_of_yes_testimonies)
        count = 0

        for cpc_city_of_yes_testimony in cpc_city_of_yes_testimonies:
            count += 1
            print(f"Processing {count} of {total}...")

            old_properties = cpc_city_of_yes_testimony.properties
            new_properties = self.__extract_data(cpc_city_of_yes_testimony).model_dump()

            # Merge old and new properties
            merged_properties = {**old_properties, **new_properties}

            # Update the testimony with merged properties
            cpc_city_of_yes_testimony.properties = merged_properties
            cpc_city_of_yes_testimony.save()

    def __extract_data(self, cpc_city_of_yes_testimony) -> TestimonyProperties:
        client = instructor.from_anthropic(
            Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY")),
            mode=instructor.Mode.ANTHROPIC_JSON,
        )

        if isinstance(cpc_city_of_yes_testimony, TranscriptChapter):
            transcript_context_chunk = self.__get_transcript_portion(
                cpc_city_of_yes_testimony
            )
        else:
            transcript_context_chunk = self.__get_transcript_portion(
                cpc_city_of_yes_testimony.testimony_chapter
            )

        transcript_context_chunk = self.__transcript_context_chunk(
            transcript_context_chunk
        )

        resp, completion = client.chat.completions.create_with_completion(
            model="claude-3-5-sonnet-20240620",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"<transcript>{transcript_context_chunk}</transcript>",
                },
            ],
            max_tokens=4096,
            response_model=TestimonyProperties,
        )

        logger.info("TRANSCRIPT:")
        logger.info(transcript_context_chunk)
        logger.info("COMPLETION:")
        logger.info(completion)
        logger.info(f"GENERATED TITLE AND SUMMARY FOR {cpc_city_of_yes_testimony.id}:")
        # logger.info(f"FOR_OR_AGAINST: {resp.for_or_against}")
        # logger.info(f"BOROUGH: {resp.borough}")
        # logger.info(f"NEIGHBORHOOD: {resp.neighborhood}")
        # logger.info(f"STATED_AFFILIATIONS: {resp.stated_affiliations}")
        logger.info(f"PRIMARY_TALKING_POINTS: {resp.primary_talking_points}")

        return resp

    def __get_transcript_portion(self, chapter: TranscriptChapter) -> Transcript:
        transcript = chapter.transcript_chapter_collection.transcript
        speakers = list(transcript.speakers.all())
        sentences = list(transcript.sentences_in_chapter(chapter))
        transcript = Transcript.from_speakers_and_sentences(speakers, sentences)
        return transcript

    def __transcript_context_chunk(self, transcript: Transcript) -> str:
        speaker_segments = [
            self.__speaker_segment_text(speaker_segment).strip()
            for speaker_segment in transcript.speaker_segments
        ]
        return "\n\n".join(speaker_segments)

    def __speaker_segment_text(self, speaker_segment: TranscriptSpeakerSegment) -> str:
        text = f"[NAME: {speaker_segment.speaker.name}] [ROLE: {speaker_segment.speaker.role}] [ORGANIZATION: {speaker_segment.speaker.organization}]\n"

        for sentence in speaker_segment.sentences:
            text += f"{sentence.text}\n"

        return text
