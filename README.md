# Every testimony at NYC City Planning's public hearing on City of Yes for Housing Opportunity

[All testimonies are viewable here on citymeetings.nyc](https://citymeetings.nyc/city-planning-commission/2024-07-10-city-of-yes-public-hearing).

Here's a post about an analysis published in this repo: [Can you figure out where people's talking points come from with LLMs?](https://vikramoberoi.com/posts/can-you-figure-out-where-peoples-talking-points-com-from-with-llms/)

## What's this dataset?

- [The transcript](https://github.com/citymeetingsnyc/cpc-city-of-yes-housing-opportunity-testimony-data/blob/main/transcript.json) of a 15-hour NYC City Planning Commission hearing on City of Yes for Housing Opportunity (YouTube: [Part 1](https://www.youtube.com/watch?v=70a3WS0l_GI), [Part 2](https://www.youtube.com/watch?v=2SMvuto6tEw))
- [211 identified speakers](https://github.com/citymeetingsnyc/cpc-city-of-yes-housing-opportunity-testimony-data/blob/main/speakers.json) from the meeting.
- [Data about all 203 testimonies](https://github.com/citymeetingsnyc/cpc-city-of-yes-housing-opportunity-testimony-data/blob/main/testimonies.json):
  - Start and end times
  - The name of the person testifying
  - Various attributes extracted from each testimony using language models
 
There's also code that demonstrates how to use LLMs to extract data from these testimonies:

- `proposal_elements_analysis.py` extracts which of the 8 elements of the zoning proposal were discussed in a given testimony.
- `talking_points_analysis.py` compares each testimony against a set of talking points to determine how aligned they are ([written about in this post](https://vikramoberoi.com/posts/can-you-figure-out-where-peoples-talking-points-com-from-with-llms/)).
- `analyze.py` is the entry point used to run these analyses (see directions below).

Example output from these analyses is in `data-examples/`

## Running the code

This project uses [poetry](https://python-poetry.org/) to manage dependencies.

First, run `poetry install` to create a virtual environment and install all the dependencies.

### Extracting elements of the proposal from each testimony

Run: `poetry run python analyze.py proposal-elements`

This will generate directory `extracted-data` in your current directory and write to it.

Each file will contain an analysis of one testimony's transcript.

See `data-examples/proposal-elements-analysis` for output from this analysis.

### Comparing talking points against each testimony

Run: `poetry run python analyze.py talking-points-analysis [TALKING_POINTS_PATH] [--stance for/against]`

You can find talking points in `talking-points/`.

If you omit --stance, it will run the comparison against all testimonies. Otherwise it will run the comparison only against testimonies for/against.

This will generate directory `extracted-data` in your current directory and write to it.

Each file will contain the output of a comparison between one testimony and the talking points.

See `data-examples/talking-points-analysis-*` for output from this analysis.

### Generating a talking points analysis report

The code above will generate a bunch of JSON files. The following command combines all of them into easier-to-read Markdown.

Run: `poetry run python analyze.py talking-points-report [EXTRACTED_DATA_DIR] [TALKING_POINTS_PATH] > report.md`

## Reading the data

The data is available in `transcript.json`, `speakers.json` and `testimony.json`:

`transcript.json` contains an object for every sentence:

```
{
"speaker": "0",
"time_marker": 0,
"start_time": 454.01694,
"end_time": 455.15662,
"text": "Good morning, everyone."
}
```

Each sentence has a speaker, which maps to a speaker label in `speakers.json`:

```
{
    "label": "0",
    "name": "Dan Garodnick",
    "role": "Chair of the City Planning Commission and Director of the Department of City Planning",
    "organization": "Department of City Planning"
}
```

You can read these into a `Transcript` object using code from `models.py`:

```
from models import Transcript

transcript = Transcript.from_speakers_and_transcript_path(
    "speakers.json",
    "transcript.json"
)
```

All testimonies are stored in `testimonies.json` as a list of objects:

```
{
    "name": "Christopher Marte",
    "citymeetings_url": "https://citymeetings.nyc/city-planning-commission/2024-07-10-city-of-yes-public-hearing/council-member-christopher-marte-on-concerns-with",
    "start_time_in_seconds": 7847.6475,
    "end_time_in_seconds": 8099.0693,
    "for_or_against": "Against",
    "borough": "Manhattan",
    "neighborhood": "Lower Manhattan",
    "stated_affiliations": [
        "NYC City Council"
    ],
    "key_points": [
        "The proposal is too large and complex for the public to fully understand in the given time frame",
        "Lack of transparency and clear communication about the proposed changes",
        "The plan will not create affordable housing",
        "Conversion of office buildings to residential will not be affordable",
        "Loss of public buildings to luxury condos",
        "Micro apartments will force families to look elsewhere",
        "Reduction in yard space, air, and light",
        "Rents will continue to rise",
        "The plan benefits those who profit from New York, not New Yorkers",
        "Need for mandated affordability instead of trickle-down approach",
        "Calls for strengthening the MIH program and requiring 100% affordability on public land",
        "Opposes changes to apartment sizes and open space regulations"
    ]
},
```

You can extract testimonies from the transcript thusly:

```
import json
from models import Transcript

transcript = Transcript.from_speakers_and_transcript_path(
    "speakers.json",
    "transcript.json"
)

with open("testimonies.json", "w") as f:
    testimonies = json.load(f)

for testimony in testimonies:
    testimony_transcript = transcript.from_start_time_to_end_time(
        testimony["start_time_in_seconds"], testimony["end_time_in_seconds"]
    )
    # Do stuff with the transcript.
```

See `analyze.py` for a concrete example.

## Contributing

If you use this data to do research or write blog post/article let me know and I'll link to it in this README.

If you'd like to contribute to this code, open a PR.

The kinds of contributions I'll accept are:

- Changes that make it easier to access the data in some way (e.g. DuckDB SQL queries for folks who don't want to write Python).
- Additional analyses of this data using LLMs.

## License

<p xmlns:cc="http://creativecommons.org/ns#" xmlns:dct="http://purl.org/dc/terms/"><a property="dct:title" rel="cc:attributionURL" href="https://github.com/citymeetingsnyc/cpc-city-of-yes-housing-opportunity-testimony-data">This dataset and code</a> by <span property="cc:attributionName">Vikram Oberoi (citymeetings.nyc)</span> is licensed under <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">CC BY-NC-SA 4.0<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/nc.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/sa.svg?ref=chooser-v1" alt=""></a></p>

- Do whatever you want with it as long as it's not commercial.
- Attribution is required.
- Please link to [citymeetings.nyc](https://citymeetings.nyc) in your work.
