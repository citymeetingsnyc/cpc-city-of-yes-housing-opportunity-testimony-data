# Every testimony at NYC City Planning's public hearing on City of Yes for Housing Opportunity

[All testimonies are viewable here on citymeetings.nyc](https://citymeetings.nyc/city-planning-commission/2024-07-10-city-of-yes-public-hearing).

This repo contains:

- The transcript of this 15-hour NYC City Planning Commission hearing on City of Yes for Housing Opportunity ([Part 1](https://www.youtube.com/watch?v=70a3WS0l_GI), [Part 2](https://www.youtube.com/watch?v=2SMvuto6tEw))
- 211 speakers identified at the meeting.
- Data about all 203 testimonies:
  - Start and end times
  - The name of the person testifying
  - Various attributes extracted from each testimony using language models
- Code that demonstrates how to extract which elements of the zoning proposal were discussed from every testimony using language models.
- The output of this code for every testimony in `extracted-data`: https://github.com/citymeetingsnyc/cpc-city-of-yes-housing-opportunity-testimony-data/tree/main/extracted-data

## Running the code

This project uses [poetry](https://python-poetry.org/) to manage dependencies.

Run:

```
poetry install
poetry run python analyze.py
```

This will generate directory `extracted-data` in your current directory and write to it.

Each file will contain an analysis of one testimony's transcript, which looks like this: https://github.com/citymeetingsnyc/cpc-city-of-yes-housing-opportunity-testimony-data/blob/main/extracted-data/adam-brodheim.json.

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

## Contributiing

If you use this data to do research or write blog post/article let me know and I'll link to it in this README.

If you'd like to contribute to this code, open a PR.

The kinds of contributions I'll accept are:

- Changes that make it easier to access the data in some way (e.g. DuckDB SQL queries for folks who don't want to write Python).
- Additional or better versions of analyses of this data using LLMs.

Questions? Email me at [vikram@citymeetings.nyc](mailto:vikram@citymeetings.nyc).

## License

<p xmlns:cc="http://creativecommons.org/ns#" xmlns:dct="http://purl.org/dc/terms/"><a property="dct:title" rel="cc:attributionURL" href="https://github.com/citymeetingsnyc/cpc-city-of-yes-housing-opportunity-testimony-data">This dataset and code</a> by <span property="cc:attributionName">Vikram Oberoi (citymeetings.nyc)</span> is licensed under <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">CC BY-NC-SA 4.0<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/nc.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/sa.svg?ref=chooser-v1" alt=""></a></p>

- Do whatever you want with it as long as it's not commercial.
- Attribution is required.
- Please link to [citymeetings.nyc](https://citymeetings.nyc) in your work.
