import json
from dataclasses import dataclass
from typing import List


@dataclass
class Speaker:
    label: str
    name: str
    role: str
    organization: str


@dataclass
class TranscriptSentence:
    speaker: Speaker
    start_time_in_seconds: float
    end_time_in_seconds: float
    time_marker: int
    text: str


@dataclass
class TranscriptSpeakerSegment:
    speaker: Speaker
    sentences: List[TranscriptSentence]


@dataclass
class Transcript:
    # These are sentences in chronological order, grouped by speaker.
    #
    # They're grouped this way because we pretty much always want to serialize them
    # thusly in prompts with minor variations:
    #
    # [SPEAKER]
    # [TIME] [TEXT]
    # [TIME] [TEXT]
    # ...
    # [SPEAKER]
    # [TIME] [TEXT]
    # [TIME] [TEXT]
    # ...
    speaker_segments: List[TranscriptSpeakerSegment]

    def from_start_time_to_end_time(
        self, start_time_in_seconds: float, end_time_in_seconds: float
    ) -> "Transcript":
        matching_sentences = []
        for segment in self.speaker_segments:
            for sentence in segment.sentences:
                if (
                    sentence.start_time_in_seconds >= start_time_in_seconds
                    and sentence.end_time_in_seconds <= end_time_in_seconds
                ):
                    matching_sentences.append(sentence)
        return Transcript.from_sentences(matching_sentences)

    @classmethod
    def from_speakers_and_transcript_path(
        cls, speakers_path: str, transcript_path: str
    ) -> "Transcript":
        with open(speakers_path, "r") as f:
            speakers = json.load(f)
        with open(transcript_path, "r") as f:
            transcript = json.load(f)
        return cls.from_speakers_and_transcript(speakers, transcript)

    @classmethod
    def from_speakers_and_transcript(
        cls, speakers: List[dict], transcript: List[dict]
    ) -> "Transcript":
        """
        {
            "speaker": "0",
            "time_marker": 0,
            "start_time": 454.01694,
            "end_time": 455.15662,
            "text": "Good morning, everyone."
        },
        """
        speaker_dict = {speaker["label"]: Speaker(**speaker) for speaker in speakers}
        sentences = []

        for sentence in transcript:
            speaker = speaker_dict[sentence["speaker"]]
            sentences.append(
                TranscriptSentence(
                    speaker,
                    sentence["start_time"],
                    sentence["end_time"],
                    sentence["time_marker"],
                    sentence["text"],
                )
            )

        return cls.from_sentences(sentences)

    @classmethod
    def from_sentences(cls, sentences: List[TranscriptSentence]) -> "Transcript":
        curr_speaker = None
        curr_speaker_segment = None
        speaker_segments = []
        for sentence in sentences:
            if sentence.speaker != curr_speaker:
                curr_speaker = sentence.speaker
                curr_speaker_segment = TranscriptSpeakerSegment(curr_speaker, [])
                speaker_segments.append(curr_speaker_segment)
            curr_speaker_segment.sentences.append(sentence)
        return cls(speaker_segments)
