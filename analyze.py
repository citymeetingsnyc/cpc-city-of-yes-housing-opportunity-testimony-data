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


@click.group()
def cli():
    """Analyze testimonies based on different criteria."""
    pass


@cli.command()
@click.option(
    "--model-provider", default="ANTHROPIC", help="Model provider (ANTHROPIC or OPENAI)"
)
@click.option("--model-name", default="claude-3-5-sonnet-20240620", help="Model name")
@click.option(
    "--stance",
    type=click.Choice(["FOR", "AGAINST"], case_sensitive=False),
    help="Filter testimonies by whether they are for or against the proposal",
)
def proposal_elements(model_provider, model_name, stance=None):
    """Analyze testimonies for which City of Yes proposal elements they discuss."""
    run_analysis(
        proposal_elements_analysis.extract,
        model_provider=model_provider,
        model_name=model_name,
        stance=stance,
    )


@cli.command()
@click.argument("reference_talking_points_path", type=click.Path(exists=True))
@click.option(
    "--model-provider", default="ANTHROPIC", help="Model provider (ANTHROPIC or OPENAI)"
)
@click.option("--model-name", default="claude-3-5-sonnet-20240620", help="Model name")
@click.option(
    "--stance",
    type=click.Choice(["FOR", "AGAINST"], case_sensitive=False),
    help="Filter testimonies by whether they are for or against the proposal",
)
def talking_points(
    reference_talking_points_path, model_provider, model_name, stance=None
):
    """Analyze testimonies for how closely they reference reference talking points.

    Talking points must be a Markdown file with bullet points, one for each talking point."""

    with open(reference_talking_points_path, "r") as f:
        reference_talking_points = f.read()

    run_analysis(
        talking_points_analysis.extract,
        reference_talking_points=reference_talking_points,
        model_provider=model_provider,
        model_name=model_name,
        stance=stance,
    )


@cli.command()
@click.argument("extracted_data_dir", type=click.Path(exists=True))
@click.argument("reference_talking_points_path", type=click.Path(exists=True))
def talking_points_report(extracted_data_dir, reference_talking_points_path):
    """Generate a report from the talking points analysis."""
    print(
        talking_points_analysis.generate_report(
            extracted_data_dir,
            reference_talking_points_path,
        )
    )


def run_analysis(
    extract_fn: Callable[[Transcript, str], BaseModel], stance=None, **extract_fn_args
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

    transcript = Transcript.from_speakers_and_transcript_path(
        SPEAKERS_PATH, TRANSCRIPT_PATH
    )

    with open(TESTIMONIES_PATH, "r") as f:
        testimonies = json.load(f)

    for testimony in testimonies:
        if stance and testimony["for_or_against"].lower() != stance.lower():
            continue

        testimony_transcript = transcript.from_start_time_to_end_time(
            testimony["start_time_in_seconds"], testimony["end_time_in_seconds"]
        )
        extracted_data = extract_fn(testimony_transcript, **extract_fn_args)

        testimony_with_extracted_data = {
            "testimony": testimony,
            "extracted_data": extracted_data.model_dump(),
        }

        name_slug = testimony["name"].replace(" ", "-").lower()
        with open(f"{EXTRACTED_DATA_DIR}/{name_slug}.json", "w") as f:
            json.dump(testimony_with_extracted_data, f, indent=4)


if __name__ == "__main__":
    cli()
