"""Find where a song starts inside a screen recording, by aligning the audio.

Why this is needed: to turn video frames into training data we must know the *song
time* at each frame, so we can look up which notes were on the highway. Two obvious
sources are not good enough - the on-screen timer is 1-second resolution, and
wall-clock timing carries the song-load latency (a second or more), which is larger
than the note windows we care about.

But the recording contains the game's own audio, and we have the source ``song.ogg``.
Cross-correlating the two locates the song within the recording to a small fraction
of a frame, using data we already capture.

Method: decode both to mono, compute short-time RMS envelopes, then normalised
cross-correlation over the envelope (not the raw waveform). Envelope correlation is
robust to the codec's phase and amplitude changes, which raw-sample correlation is
not.
"""
from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SR = 16000          # sample rate for decoding (plenty for an envelope)
HOP_MS = 10.0       # envelope hop -> 100 Hz, finer than a 30 fps video frame
WIN_MS = 40.0       # envelope window


@dataclass
class Alignment:
    offset_seconds: float   # where the song starts, in the recording's timeline
    score: float            # normalised correlation at the best offset (1.0 = perfect)
    peak_ratio: float       # best peak / second-best distinct peak (confidence)

    @property
    def confident(self) -> bool:
        return self.score > 0.5 and self.peak_ratio > 1.5


def decode_mono(path: str | Path, sr: int = SR) -> np.ndarray:
    """Decode any audio/video file to a mono float32 array at `sr`."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav = Path(tmp.name)

    try:
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-i", str(path), "-ac", "1", "-ar", str(sr), "-f", "wav", str(wav)],
            check=True,
        )
        return _read_wav_mono(wav)
    finally:
        wav.unlink(missing_ok=True)


def _read_wav_mono(path: Path) -> np.ndarray:
    """Minimal WAV reader for ffmpeg's PCM output (avoids a scipy/soundfile dep)."""
    import wave

    with wave.open(str(path), "rb") as w:
        frames = w.readframes(w.getnframes())
        width = w.getsampwidth()
        channels = w.getnchannels()

    if width == 2:
        data = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        data = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    elif width == 1:
        data = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"unsupported sample width {width}")

    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data


def envelope(x: np.ndarray, sr: int = SR, hop_ms: float = HOP_MS, win_ms: float = WIN_MS) -> np.ndarray:
    """Short-time RMS envelope, mean-removed."""
    hop = max(1, int(sr * hop_ms / 1000.0))
    win = max(hop, int(sr * win_ms / 1000.0))
    n = max(0, 1 + (len(x) - win) // hop) if len(x) >= win else 0
    if n == 0:
        return np.zeros(1, dtype=np.float32)

    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    frames = x[idx]
    env = np.sqrt((frames ** 2).mean(axis=1))
    return (env - env.mean()).astype(np.float32)


def onset_envelope(x: np.ndarray, sr: int = SR, n_fft: int = 1024, hop_ms: float = HOP_MS) -> np.ndarray:
    """Spectral-flux onset strength, mean-removed.

    Loudness (RMS) alignment does NOT work for our synthetic test songs: they have a
    near-constant amplitude envelope, so there is no structure to lock onto. The note
    *onsets* are distinctive even at constant volume, and spectral flux (the positive
    change in magnitude across frequency bins between frames) exposes exactly those.
    """
    hop = max(1, int(sr * hop_ms / 1000.0))
    if len(x) < n_fft:
        return np.zeros(1, dtype=np.float32)

    n = 1 + (len(x) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n)[:, None]
    frames = x[idx] * np.hanning(n_fft)[None, :]

    mag = np.abs(np.fft.rfft(frames, axis=1))
    flux = np.maximum(0.0, mag[1:] - mag[:-1]).sum(axis=1)

    # Local mean removal keeps it a zero-mean signal for correlation.
    if flux.size > 4:
        kernel = np.ones(5) / 5.0
        flux = flux - np.convolve(flux, kernel, mode="same")
    return flux.astype(np.float32)


#: name -> function used to turn audio into the signal that gets correlated
FEATURES = {
    "onset": onset_envelope,
    "rms": lambda x, sr=SR: envelope(x, sr),
    "raw": None,  # handled directly in find_offset via FFT (needs the waveform)
}


def _fft_correlate(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalised sliding dot product of `b` against every window of `a`.

    Returns (normalised correlation per offset, raw dot products). FFT-based so a
    whole-song search over a minute of 16 kHz audio is fast. Unlike the envelope and
    spectral-flux features this keeps phase, which matters because the game's mix is
    the same waveform as song.ogg - only resampled and mixed with effects - so
    sample-level structure survives.
    """
    if len(b) >= len(a):
        raise ValueError("need the search signal to be shorter than the recording")

    n = len(a) + len(b) - 1
    nfft = 1 << (n - 1).bit_length()
    corr = np.fft.irfft(np.fft.rfft(a, nfft) * np.conj(np.fft.rfft(b, nfft)), nfft)[:len(a) - len(b) + 1]

    # Energy of each window of `a` (O(n) via a prefix sum), for normalisation.
    prefix = np.concatenate([[0.0], np.cumsum(a.astype(np.float64) ** 2)])
    energy = prefix[len(b):] - prefix[:len(prefix) - len(b)]
    energy = np.sqrt(np.maximum(energy, 1e-12))

    norm_b = max(float(np.sqrt((b.astype(np.float64) ** 2).sum())), 1e-12)
    return corr / (energy * norm_b), corr


def find_offset(
    recording: str | Path,
    song: str | Path,
    *,
    min_offset: float = 0.0,
    max_offset: float | None = None,
    sr: int = SR,
    feature: str = "onset",
) -> Alignment:
    """Locate `song` inside `recording`; returns the offset in seconds.

    ``feature`` selects what is correlated: "onset" (spectral flux, default - works on
    constant-loudness material) or "rms" (loudness envelope).
    """
    if feature not in FEATURES:
        raise ValueError(f"unknown feature {feature!r}; expected one of {sorted(FEATURES)}")

    if feature == "raw":
        rec_x = decode_mono(recording, sr)
        sng_x = decode_mono(song, sr)
        # Remove DC: a constant offset in the sink capture would otherwise dominate.
        rec_x = rec_x - rec_x.mean()
        sng_x = sng_x - sng_x.mean()

        if len(sng_x) >= len(rec_x):
            raise ValueError("song is longer than the recording; nothing to search")

        norm, _ = _fft_correlate(rec_x, sng_x)
        lo = max(0, int(min_offset * sr))
        hi = len(rec_x) - len(sng_x)
        if max_offset is not None:
            hi = min(hi, int(max_offset * sr))
        if hi <= lo:
            raise ValueError("empty search range for the song offset")

        window = norm[lo:hi + 1]
        best = int(np.argmax(window))
        score = float(window[best])

        guard = max(1, len(sng_x) // 2)
        far = window.copy()
        far[max(0, best - guard):best + guard] = -np.inf
        second = float(np.max(far)) if np.isfinite(far).any() else 0.0
        ratio = float(window[best] / second) if second > 0 else float("inf")

        return Alignment(offset_seconds=(best + lo) / sr, score=score, peak_ratio=ratio)

    to_feature = FEATURES[feature]

    rec = to_feature(decode_mono(recording, sr), sr)
    sng = to_feature(decode_mono(song, sr), sr)

    hop_seconds = HOP_MS / 1000.0
    if len(sng) >= len(rec):
        raise ValueError("song is longer than the recording; nothing to search")

    # Search only offsets that fit the song inside the recording.
    lo = max(0, int(min_offset / hop_seconds))
    hi = len(rec) - len(sng)
    if max_offset is not None:
        hi = min(hi, int(max_offset / hop_seconds))
    if hi <= lo:
        raise ValueError("empty search range for the song offset")

    window = rec[lo:hi + len(sng)]
    corr = np.correlate(window, sng, mode="valid")

    # Normalise by the local energy so a loud section cannot dominate.
    energy = np.sqrt(np.convolve(window ** 2, np.ones(len(sng)), mode="valid"))
    energy[energy == 0] = 1e-9
    norm = corr / energy

    best = int(np.argmax(norm))
    score = float(norm[best] / max(np.sqrt((sng ** 2).sum()), 1e-9))

    # Confidence: the best peak vs the best peak far away from it.
    guard = max(1, len(sng) // 2)
    far = norm.copy()
    far[max(0, best - guard):best + guard] = -np.inf
    second = float(np.max(far)) if np.isfinite(far).any() else 0.0
    peak_ratio = float(norm[best] / second) if second > 0 else float("inf")

    return Alignment(
        offset_seconds=(best + lo) * hop_seconds,
        score=score,
        peak_ratio=peak_ratio,
    )


def _main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Align a song inside a screen recording")
    ap.add_argument("--recording", required=True)
    ap.add_argument("--song", required=True, help="path to the song's audio (song.ogg)")
    ap.add_argument("--max-offset", type=float, default=None)
    ap.add_argument("--feature", default="onset", choices=sorted(FEATURES),
                    help="what to correlate (default: onset, needed for constant-loudness songs)")
    args = ap.parse_args()

    a = find_offset(args.recording, args.song, max_offset=args.max_offset, feature=args.feature)
    print(f"feature        : {args.feature}")
    print(f"song starts at : {a.offset_seconds:.3f} s into the recording")
    print(f"correlation    : {a.score:.3f}")
    print(f"peak ratio     : {a.peak_ratio:.2f}")
    print(f"confident      : {a.confident}")
    if not a.confident:
        print("\nWARNING: low confidence - check the recording actually contains the song "
              "audio, and that the envelope is not dominated by silence.")
    return 0 if a.confident else 1


if __name__ == "__main__":
    raise SystemExit(_main())
