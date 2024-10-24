"""
This script analyzes a transcript of testimonies regarding the NYC City of Yes for Housing Opportunity proposal.

It determines whether each speaker is for or against the proposal based on their testimony, extracts relevant information such as borough, neighborhood, stated affiliations, and elements discussed, and outputs the analysis in a structured JSON format.

Usage:
    poetry run python for_or_against.py --transcript path/to/transcript.json --output path/to/output.json --limit N

Arguments:
    --transcript: Path to the transcript JSON file containing the testimonies.
    --output: Path to save the analysis results in JSON format.
    --limit: (Optional) Maximum number of speakers to analyze.
"""

from typing import List, Dict, Optional, Literal
from datetime import datetime
from pydantic import BaseModel, Field
from anthropic import Anthropic
import instructor
import json
import os


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
    borough: Literal[
        "Manhattan", "Queens", "Brooklyn", "Bronx", "Staten Island", "Unknown"
    ] = Field(
        description="The borough in which the individual giving testimony lives. If it is not clear in which borough the individual lives, use 'Unknown'. If you can extrapolate the borough the individual lives in from the neighborhood they stated that they live in or the community board that they are in, use that."
    )
    neighborhood: str = Field(
        description="The neighborhood in which the individual giving testimony lives. If it is not clear in which neighborhood the individual lives, use 'Unknown'."
    )
    stated_affiliations: List[str] = Field(
        description="""
        The affiliations that the individual giving testimony has stated that they have.

        An affiliation can include an organization/association or government body they are a part of.

        An affiliation does not include where the individual lives, or their role.

        Affiliations are purely entities that the individual states they are a part of.

        This value should be a list of these affiliations.
        """
    )
    elements_discussed: List[str] = Field(
        description="""
        The elements of the City of Yes for Housing Opportunity proposal that the individual discussed in their testimony.

        Possible elements include:
        - Universal Affordability Preference (UAP)
        - Residential Conversions
        - Town Center Zoning
        - Removing Parking Mandates
        - Accessory Dwelling Units (ADUs)
        - Transit-Oriented Development
        - Campuses
        - Small and Shared Housing

        If none of these elements are discussed, use an empty list.
        """
    )


class AnalysisResult(BaseModel):
    speaker_id: str
    testimony: str
    analysis: ForOrAgainstStance
    start_time: float = Field(
        description="The first time the speaker appears in the transcript, in seconds."
    )
    timestamp: str


class AnalysisOutput(BaseModel):
    metadata: Dict = Field(
        default_factory=lambda: {
            "analysis_date": datetime.now().isoformat(),
            "version": "1.1"
        }
    )
    results: List[AnalysisResult]
    summary: Dict[str, int] = Field(
        default_factory=lambda: {"for": 0, "against": 0, "total_analyzed": 0}
    )


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

If the testimony mentions where the individual lives, identify the borough in which they reside. The possible boroughs are:
- Manhattan
- Queens
- Brooklyn
- Bronx
- Staten Island
If you cannot determine the borough, mark it as 'Unknown'. If the neighborhood or community board provides clues, use that information to determine the borough.

Additionally, extract the following information:
- Neighborhood: Identify the neighborhood where the speaker resides. If unclear, use 'Unknown'.
- Stated Affiliations: List any organizations, associations, or government bodies the speaker is affiliated with, as mentioned in their testimony.
- Elements Discussed: List the specific elements of the City of Yes proposal that the speaker discusses in their testimony.

Ensure that all extracted information is accurate and based solely on the testimony provided.
"""


def analyze_transcript_stances(transcript_path: str, limit: Optional[int] = None) -> AnalysisOutput:
    """
    Analyze transcript stances and return structured JSON output.

    Args:
        transcript_path (str): Path to the transcript JSON file
        limit (int, optional): Maximum number of speakers to analyze

    Returns:
        AnalysisOutput: Structured analysis results
    """
    client = instructor.from_anthropic(
        Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    )

    with open(transcript_path, 'r') as f:
        transcript = json.load(f)

    speaker_messages = {}
    speaker_start_times = {}
    for entry in transcript:
        speaker = entry['speaker']
        message_start_time = entry.get('start_time')
        if message_start_time is None:
            # If 'start_time' is missing, default to 0.0 or handle as needed
            message_start_time = 0.0

        if speaker not in speaker_messages:
            speaker_messages[speaker] = []
            speaker_start_times[speaker] = message_start_time
        else:
            # Update start_time only if the current message's start_time is earlier
            if message_start_time < speaker_start_times[speaker]:
                speaker_start_times[speaker] = message_start_time

        speaker_messages[speaker].append(entry)

    if limit:
        speakers_to_analyze = list(speaker_messages.keys())[:limit]
        speaker_messages = {k: speaker_messages[k] for k in speakers_to_analyze}
        speaker_start_times = {k: speaker_start_times[k] for k in speakers_to_analyze}

    analysis_output = AnalysisOutput(results=[])

    total_speakers = len(speaker_messages)
    print(f"Total speakers to analyze: {total_speakers}")

    for idx, (speaker, messages) in enumerate(speaker_messages.items(), start=1):
        testimony = " ".join(msg['text'] for msg in messages)

        if len(testimony.split()) < 10:
            print(f"Skipping speaker {speaker} due to insufficient testimony length.")
            continue

        start_time = speaker_start_times.get(speaker, 0.0)
        print(f"Analyzing speaker {idx}/{total_speakers} - ID: {speaker} (Start Time: {start_time} seconds)...")

        try:
            kwargs = {
                "response_model": ForOrAgainstStance,
                "model": "claude-3-5-sonnet-20241022",  # Newest version of Claude
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Please analyze this testimony to determine if the speaker is for or against the City of Yes proposal, identify their borough and neighborhood, list any stated affiliations, and enumerate the elements of the proposal they discussed:\n\n{testimony}"
                    },
                ],
                "max_tokens": 8192
            }

            response, _ = client.chat.completions.create_with_completion(**kwargs)

            # Ensure that all required fields are present
            analysis = response.dict()
            missing_fields = [field for field in ForOrAgainstStance.model_fields if field not in analysis]
            if missing_fields:
                raise ValueError(f"Missing fields in response: {missing_fields}")

            stance = ForOrAgainstStance(**analysis)

            result = AnalysisResult(
                speaker_id=speaker,
                testimony=testimony,
                analysis=stance,
                start_time=start_time,
                timestamp=datetime.now().isoformat()
            )

            analysis_output.results.append(result)

            # Update summary counts
            if stance.for_or_against.lower() == "for":
                analysis_output.summary["for"] += 1
            elif stance.for_or_against.lower() == "against":
                analysis_output.summary["against"] += 1
            analysis_output.summary["total_analyzed"] += 1

        except Exception as e:
            print(f"Error analyzing speaker {speaker}: {str(e)}")

    return analysis_output


def save_analysis(analysis: AnalysisOutput, output_path: str = "stance_analysis_results.json"):
    """Save analysis results to a JSON file."""
    with open(output_path, 'w') as f:
        json.dump(analysis.model_dump(), f, indent=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Analyze transcript stances')
    parser.add_argument('--transcript', default='transcript.json', help='Path to transcript JSON file')
    parser.add_argument('--limit', type=int, help='Number of speakers to analyze')
    parser.add_argument('--output', default='stance_analysis_results.json', help='Output JSON file path')

    args = parser.parse_args()

    try:
        analysis_results = analyze_transcript_stances(args.transcript, args.limit)
        save_analysis(analysis_results, args.output)
        print(f"\nAnalysis complete. Results saved to {args.output}")
        print("\nSummary:")
        print(json.dumps(analysis_results.summary, indent=2))
    except Exception as e:
        print(f"An error occurred: {str(e)}")
