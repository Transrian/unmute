"""Audio codec utilities for OpenAI-compatible /v1/audio endpoints.

Decode uploaded audio files to mono float32 PCM at 24 kHz,
and encode PCM back to mp3 / wav / ogg / flac.
"""

from io import BytesIO
from logging import getLogger

import av
import numpy as np

logger = getLogger(__name__)

# Supported input formats (file extensions → av format names)
INPUT_FORMATS = {
    "flac": "flac",
    "m4a": "ipod",
    "mp3": "mp3",
    "mp4": "mp4",
    "ogg": "ogg",
    "wav": "wav",
    "webm": "webm",
}

# Supported output formats for /v1/audio/speech
OUTPUT_FORMATS = {
    "mp3": {"av_format": "mp3", "codec": "libmp3lame", "mime": "audio/mpeg"},
    "wav": {"av_format": "wav", "codec": "pcm_s16le", "mime": "audio/wav"},
    "flac": {"av_format": "flac", "codec": "flac", "mime": "audio/flac"},
    "ogg": {"av_format": "ogg", "codec": "libvorbis", "mime": "audio/ogg"},
}


def guess_audio_format(filename: str) -> str | None:
    """Guess the audio format from the filename extension."""
    if "." not in filename:
        return None
    ext = filename.rsplit(".", 1)[-1].lower()
    return INPUT_FORMATS.get(ext)


def decode_audio_to_pcm(
    data: bytes,
    filename: str = "audio.wav",
    target_sample_rate: int = 24000,
) -> np.ndarray:
    """Decode an audio file to mono float32 PCM at the target sample rate.

    Args:
        data: Raw bytes of the audio file.
        filename: Original filename (used to guess format).
        target_sample_rate: Output sample rate (default 24000).

    Returns:
        1D numpy array of float32 samples in [-1, 1].
    """
    fmt = guess_audio_format(filename)
    buf = BytesIO(data)

    container = av.open(buf, format=fmt)

    try:
        stream = next(s for s in container.streams if s.type == "audio")
        resampler = av.AudioResampler(
            format="s16", layout="mono", rate=target_sample_rate
        )

        all_arrays: list[np.ndarray] = []
        for frame in container.decode(stream):
            resampled = resampler.resample(frame)
            for f in resampled:
                arr = f.to_ndarray()[0]  # mono → 1D
                all_arrays.append(arr)

        if not all_arrays:
            raise ValueError("No audio samples decoded from the input file.")

        pcm_int16 = np.concatenate(all_arrays)
        # Convert int16 to float32 in [-1, 1]
        return pcm_int16.astype(np.float32) / 32768.0

    finally:
        container.close()


def encode_pcm_to_audio(
    pcm: np.ndarray,
    format: str = "mp3",
    sample_rate: int = 24000,
) -> tuple[bytes, str]:
    """Encode float32 PCM to an audio file format.

    Args:
        pcm: 1D numpy array of float32 samples in [-1, 1].
        format: Output format — one of "mp3", "wav", "flac", "ogg".
        sample_rate: Sample rate of the PCM data.

    Returns:
        Tuple of (encoded bytes, mime_type).
    """
    if format not in OUTPUT_FORMATS:
        raise ValueError(
            f"Unsupported output format '{format}'. Choose from: {list(OUTPUT_FORMATS.keys())}"
        )

    config = OUTPUT_FORMATS[format]

    # Convert float32 to int16
    samples = np.clip(pcm, -1.0, 1.0)
    samples_int16 = (samples * 32767).astype(np.int16)

    buf = BytesIO()
    container = av.open(buf, "w", format=config["av_format"])
    stream = container.add_stream(config["codec"], rate=sample_rate, layout="mono")

    # Use explicit frame creation to avoid from_ndarray planar format issues
    frame = av.AudioFrame(format="s16", layout="mono", samples=len(samples_int16))
    frame.sample_rate = sample_rate
    frame.planes[0].update(samples_int16.tobytes())

    for packet in stream.encode(frame):
        container.mux(packet)
    # Flush encoder
    for packet in stream.encode():
        container.mux(packet)
    container.close()

    return buf.getvalue(), config["mime"]
