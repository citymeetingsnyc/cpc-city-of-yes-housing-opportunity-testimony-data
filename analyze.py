import json
import logging
import os

import click
from dotenv import load_dotenv
from rich.logging import RichHandler
from rich.prompt import Confirm

import city_of_yes_elements
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


@click.command()
@click.argument(
    "analysis", type=click.Choice(["coy-proposal-elements", "reference-talking-points"])
)
def main(analysis):
    """
    Analyze testimonies based on the specified analysis type.
    """
    if analysis == "coy-proposal-elements":
        extract_fn = city_of_yes_elements.extract
    elif analysis == "reference-talking-points":
        extract_fn = reference_talking_points.extract
    else:
        raise ValueError(f"Unknown analysis type: {analysis}")

    logger.info(f"Running analysis: {analysis}")

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
        testimony_transcript = transcript.from_start_time_to_end_time(
            testimony["start_time_in_seconds"], testimony["end_time_in_seconds"]
        )
        extracted_data = extract_fn(testimony_transcript)

        testimony_with_extracted_data = {
            "testimony": testimony,
            "extracted_data": extracted_data,
        }

        name_slug = testimony["name"].replace(" ", "-").lower()
        with open(f"{EXTRACTED_DATA_DIR}/{name_slug}.json", "w") as f:
            json.dump(testimony_with_extracted_data, f, indent=4)


if __name__ == "__main__":
    main()
