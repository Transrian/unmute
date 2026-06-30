from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from ruamel.yaml import YAML

from unmute.llm.system_prompt import Instructions


class SoundInstance(BaseModel):
    id: int
    name: str
    username: str
    license: str


class FreesoundVoiceSource(BaseModel):
    source_type: Literal["freesound"] = "freesound"
    url: str
    start_time: float = 0
    sound_instance: SoundInstance
    path_on_server: str


class FileVoiceSource(BaseModel):
    source_type: Literal["file"] = "file"
    path_on_server: str
    description: str | None = None
    description_link: str | None = None


class VoiceSample(BaseModel):
    model_config = {"extra": "forbid"}

    name: str | None = None
    comment: str | None = None
    good: bool | None = None
    instructions: Instructions | None = None
    source: FreesoundVoiceSource | FileVoiceSource = Field(discriminator="source_type")


class VoiceList:
    def __init__(self):
        self.path = Path(__file__).parents[1] / "voices.yaml"
        with self.path.open() as f:
            self.voices = [VoiceSample(**sound) for sound in YAML().load(f)]

    def save(self):
        with self.path.open("w") as f:
            yaml = YAML()
            yaml.width = float("inf")  # Disable line wrapping

            # Put "good" voices first, then undecided, then bad.
            # The sort is stable, so the order is otherwise preserved
            voices = sorted(
                self.voices, key=lambda x: {True: 0, None: 1, False: 2}[x.good]
            )
            yaml.dump(
                [
                    voice.model_dump(
                        # This would also exclude the discriminator field :(
                        # exclude_defaults=True,
                        exclude_none=True,
                        exclude={"source": {"sound_instance": ["previews"]}},  # type: ignore
                    )
                    for voice in voices
                ],
                f,
            )
