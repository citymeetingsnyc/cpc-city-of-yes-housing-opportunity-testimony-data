from models import Transcript


def serialize_transcript(transcript: Transcript) -> str:
    text = ""
    for speaker_segment in transcript.speaker_segments:
        speaker = speaker_segment.speaker
        text += f"[NAME: {speaker.name} ROLE: {speaker.role} ORGANIZATION: {speaker.organization}]\n"
        for sentence in speaker_segment.sentences:
            text += f"{sentence.text}\n"
        text += "\n"
    return text.strip()
