import json
import logging
import os
from typing import Callable

import click
from dotenv import load_dotenv
from pydantic import BaseModel
from rich.logging import RichHandler
from rich.prompt import Confirm

import proposal_elements_analysis
import talking_points_analysis
from models import Transcript
import for_or_against  # Testing the imports

FORMAT = "%(message)s"
logging.basicConfig(
    level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)
logger = logging.getLogger(__name__)

load_dotenv()

SPEAKERS_PATH = "speakers.json"
TRANSCRIPT_PATH = "transcript.json"
TESTIMONIES_PATH = "testimonies.json"
EXTRACTED_DATA_DIR = "extracted-data"


def speakers_path(source_data_dir: str) -> str:
    return f"{source_data_dir}/speakers.json"


def transcript_path(source_data_dir: str) -> str:
    return f"{source_data_dir}/transcript.json"


def testimonies_path(source_data_dir: str) -> str:
    return f"{source_data_dir}/testimonies.json"


@click.group()
def cli():
    """Analyze testimonies based on different criteria."""
    pass


@cli.command()
@click.argument("source_data_dir", type=click.Path(exists=True))
@click.option(
    "--model-provider", default="ANTHROPIC", help="Model provider (ANTHROPIC or OPENAI)"
)
@click.option("--model-name", default="claude-3-5-sonnet-20241022", help="Model name")
@click.option(
    "--stance",
    type=click.Choice(["FOR", "AGAINST"], case_sensitive=False),
    help="Filter testimonies by whether they are for or against the proposal",
)
def proposal_elements(source_data_dir, model_provider, model_name, stance=None):
    """Analyze testimonies for which City of Yes proposal elements they discuss."""
    run_analysis(
        proposal_elements_analysis.extract,
        source_data_dir=source_data_dir,
        model_provider=model_provider,
        model_name=model_name,
        stance=stance,
    )


@cli.command()
@click.argument("source_data_dir", type=click.Path(exists=True))
@click.argument("reference_talking_points_path", type=click.Path(exists=True))
@click.option(
    "--model-provider", default="ANTHROPIC", help="Model provider (ANTHROPIC or OPENAI)"
)
@click.option("--model-name", default="claude-3-5-sonnet-20241022", help="Model name") # Newest verison of Claude, as of October
@click.option(
    "--stance",
    type=click.Choice(["FOR", "AGAINST"], case_sensitive=False),
    help="Filter testimonies by whether they are for or against the proposal",
)
def talking_points(
    source_data_dir,
    reference_talking_points_path,
    model_provider,
    model_name,
    stance=None,
):
    """Analyze testimonies for how closely they reference reference talking points.

    Talking points must be a Markdown file with bullet points, one for each talking point."""

    with open(reference_talking_points_path, "r") as f:
        reference_talking_points = f.read()

    run_analysis(
        talking_points_analysis.extract,
        source_data_dir=source_data_dir,
        reference_talking_points=reference_talking_points,
        model_provider=model_provider,
        model_name=model_name,
        stance=stance,
    )


@cli.command()
@click.argument("source_data_dir", type=click.Path(exists=True))
@click.argument("extracted_data_dir", type=click.Path(exists=True))
@click.argument("reference_talking_points_path", type=click.Path(exists=True))
def talking_points_report(
    source_data_dir, extracted_data_dir, reference_talking_points_path
):
    """Generate a report from the talking points analysis."""
    print(
        talking_points_analysis.generate_report(
            source_data_dir,
            extracted_data_dir,
            reference_talking_points_path,
        )
    )

@cli.command() # NEW - this should hopefully take in for_or_against.py and run it easily 
@click.option(
    "--model-provider", default="ANTHROPIC", help="Model provider (ANTHROPIC or OPENAI)"
)
@click.option("--model-name", default="claude-3-5-sonnet-20241022", help="Model name")
def for_against(model_provider, model_name):
    """Analyze testimonies to determine if they are for or against the proposal."""
    run_analysis(
        for_or_against.extract,
        model_provider=model_provider,
        model_name=model_name,
    )

def run_analysis(
    extract_fn: Callable[[Transcript, str], BaseModel],
    source_data_dir=None,
    stance=None,
    **extract_fn_args,
):
    if not os.path.exists(EXTRACTED_DATA_DIR):
        os.makedirs(EXTRACTED_DATA_DIR)

    if os.listdir(EXTRACTED_DATA_DIR):
        overwrite = Confirm.ask(
            "Files already exist in the extracted-data directory. Overwrite them?"
        )
        if not overwrite:
            logger.info("Operation cancelled. Existing files will not be overwritten.")
            return

    # Load the transcript and testimonies
    transcript = Transcript.from_speakers_and_transcript_path(
        speakers_path(source_data_dir), transcript_path(source_data_dir)
    )

    with open(testimonies_path(source_data_dir), "r") as f:
        testimonies = json.load(f)

    # Prompt the user to process a certain number of testimonies or all
    user_input = input("Enter the number of testimonies to process (or type 'all' to process all): ")

    if user_input.lower() == "all":
        num_to_process = len(testimonies)
    else:
        try:
            num_to_process = int(user_input)
            if num_to_process <= 0:
                raise ValueError("Number must be greater than zero.")
        except ValueError as e:
            print(f"Invalid input: {e}")
            return

    # Process the testimonies, limited by the user-specified count
    processed_count = 0
    for idx, testimony in enumerate(testimonies):
        if idx >= num_to_process:
            break

        if stance and testimony["for_or_against"].lower() != stance.lower():
            continue

        # Get the testimony transcript
        testimony_transcript = transcript.from_start_time_to_end_time(
            testimony["start_time_in_seconds"], testimony["end_time_in_seconds"]
        )
        # Extract data from the testimony
        extracted_data = extract_fn(testimony_transcript, **extract_fn_args)

        # Combine the testimony and extracted data
        testimony_with_extracted_data = {
            "testimony": testimony,
            "extracted_data": extracted_data.model_dump(),
        }

        # Create a file name based on the person's name
        name_slug = testimony["name"].replace(" ", "-").lower()

        # Save the extracted data to a JSON file
        with open(f"{EXTRACTED_DATA_DIR}/{name_slug}.json", "w") as f:
            json.dump(testimony_with_extracted_data, f, indent=4)

        processed_count += 1

    print(f"Processed {processed_count} testimonies.")


if __name__ == "__main__":
    cli()
