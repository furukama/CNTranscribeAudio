from __future__ import annotations

import logging
import os
import re
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import jieba
import torch
import yt_dlp
from pypinyin import Style, pinyin
from qwen_asr import Qwen3ASRModel
from transformers import pipeline

ProgressCallback = Callable[[int, str], None]

jieba.setLogLevel(logging.WARNING)
jieba.initialize()

SILENCE_DETECT_NOISE_DB = "-35dB"
SILENCE_DETECT_MIN_DURATION_SECONDS = 0.45
MIN_SPEECH_SEGMENT_SECONDS = 0.35


@dataclass
class WordToken:
    word: str
    pinyin: str
    english: str


@dataclass
class SegmentOutput:
    start: float | None
    end: float | None
    chinese: str
    tokens: list[WordToken]
    full_english: str


class ChineseTranscriptionPipeline:
    def __init__(self) -> None:
        self._logger = logging.getLogger("cntranscribe.pipeline")
        self._asr_models: dict[tuple[str, str], Qwen3ASRModel] = {}
        self._asr_has_aligner: dict[tuple[str, str], bool] = {}
        self._hf_translator = None
        self._mlx_models: dict[str, tuple[Any, Any, Any]] = {}

    def _load_asr_model(self, asr_model_name: str, aligner_name: str) -> tuple[Qwen3ASRModel, bool]:
        key = (asr_model_name, aligner_name)
        if key in self._asr_models:
            return self._asr_models[key], self._asr_has_aligner[key]

        if torch.cuda.is_available():
            dtype = torch.bfloat16
            device_map: str = "cuda:0"
        else:
            dtype = torch.float32
            device_map = "cpu"

        self._logger.info("Loading Qwen ASR model: %s", asr_model_name)
        try:
            model = Qwen3ASRModel.from_pretrained(
                asr_model_name,
                dtype=dtype,
                device_map=device_map,
                max_new_tokens=1024,
                forced_aligner=aligner_name,
                forced_aligner_kwargs={
                    "dtype": dtype,
                    "device_map": device_map,
                },
            )
            has_aligner = True
        except Exception as exc:
            # Some converted variants may not support the aligner path.
            self._logger.warning("Could not load aligner (%s). Retrying ASR model without aligner.", exc)
            model = Qwen3ASRModel.from_pretrained(
                asr_model_name,
                dtype=dtype,
                device_map=device_map,
                max_new_tokens=1024,
            )
            has_aligner = False

        self._asr_models[key] = model
        self._asr_has_aligner[key] = has_aligner
        return model, has_aligner

    def _load_hf_translator(self):
        if self._hf_translator is None:
            device = 0 if torch.cuda.is_available() else -1
            self._hf_translator = pipeline(
                "translation",
                model="Helsinki-NLP/opus-mt-zh-en",
                device=device,
            )
        return self._hf_translator

    def _load_mlx(self, model_name: str) -> tuple[Any, Any, Any]:
        if model_name in self._mlx_models:
            return self._mlx_models[model_name]

        from mlx_lm import generate, load  # type: ignore

        model, tokenizer = load(model_name)
        loaded = (model, tokenizer, generate)
        self._mlx_models[model_name] = loaded
        return loaded

    @staticmethod
    def _download_youtube_audio(url: str, output_wav: Path) -> Path:
        outtmpl = str(output_wav.with_suffix(""))
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": outtmpl,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                    "preferredquality": "192",
                }
            ],
            "quiet": True,
            "no_warnings": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        if output_wav.exists():
            return output_wav

        matches = sorted(output_wav.parent.glob(f"{output_wav.stem}*.wav"))
        if not matches:
            raise RuntimeError("Audio download failed: no WAV file created.")
        return matches[0]

    @staticmethod
    def _trim_audio_seconds(source_wav: Path, seconds: int) -> Path:
        trimmed = source_wav.with_name(f"{source_wav.stem}.debug{seconds}.wav")
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(source_wav),
            "-t",
            str(seconds),
            "-acodec",
            "copy",
            str(trimmed),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"Failed to trim audio for DEBUG mode: {proc.stderr.strip()}")
        return trimmed

    @staticmethod
    def _probe_audio_duration_seconds(audio_path: Path) -> float | None:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            return None
        try:
            value = float(proc.stdout.strip())
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None

    @staticmethod
    def _normalize_qwen_segments(result_item: Any) -> list[dict[str, Any]]:
        if isinstance(result_item, dict):
            maybe_segments = result_item.get("segments")
            if isinstance(maybe_segments, list):
                return maybe_segments

        for attr_name in ("segments", "sentences", "chunks", "timestamps", "time_stamps"):
            if hasattr(result_item, attr_name):
                maybe_segments = getattr(result_item, attr_name)
                if isinstance(maybe_segments, list):
                    normalized: list[dict[str, Any]] = []
                    for seg in maybe_segments:
                        if isinstance(seg, dict):
                            normalized.append(seg)
                        else:
                            normalized.append({"text": str(seg)})
                    return normalized

        text = ""
        if isinstance(result_item, dict):
            text = str(result_item.get("text") or "").strip()
        elif hasattr(result_item, "text"):
            text = str(getattr(result_item, "text") or "").strip()

        return [{"text": text, "start": None, "end": None}] if text else []

    @staticmethod
    def _split_text_sentences(text: str) -> list[str]:
        text = (text or "").strip()
        if not text:
            return []
        parts = re.findall(r"[^。！？!?；;]+[。！？!?；;]?", text)
        cleaned = [part.strip() for part in parts if part and part.strip()]
        return cleaned if cleaned else [text]

    @staticmethod
    def _detect_pause_segments(audio_path: Path, duration_seconds: float | None) -> list[tuple[float, float]]:
        if duration_seconds is None or duration_seconds <= 0:
            return []

        cmd = [
            "ffmpeg",
            "-i",
            str(audio_path),
            "-af",
            f"silencedetect=noise={SILENCE_DETECT_NOISE_DB}:d={SILENCE_DETECT_MIN_DURATION_SECONDS}",
            "-f",
            "null",
            "-",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        log = f"{proc.stdout}\n{proc.stderr}"

        starts = [float(v) for v in re.findall(r"silence_start:\s*([0-9.]+)", log)]
        ends = [float(v) for v in re.findall(r"silence_end:\s*([0-9.]+)", log)]
        if not starts or not ends:
            return []

        silence_ranges: list[tuple[float, float]] = []
        pair_count = min(len(starts), len(ends))
        for i in range(pair_count):
            s = max(0.0, starts[i])
            e = min(duration_seconds, ends[i])
            if e > s:
                silence_ranges.append((s, e))

        if not silence_ranges:
            return []

        speech_ranges: list[tuple[float, float]] = []
        cursor = 0.0
        for s, e in silence_ranges:
            if s - cursor >= MIN_SPEECH_SEGMENT_SECONDS:
                speech_ranges.append((cursor, s))
            cursor = max(cursor, e)
        if duration_seconds - cursor >= MIN_SPEECH_SEGMENT_SECONDS:
            speech_ranges.append((cursor, duration_seconds))
        return speech_ranges

    @staticmethod
    def _segments_from_sentences_and_ranges(
        sentences: list[str],
        ranges: list[tuple[float, float]],
        duration_seconds: float | None,
    ) -> list[dict[str, Any]]:
        if not sentences:
            return []

        if not ranges:
            # Fallback to proportional splitting over known duration.
            duration = duration_seconds or float(max(20, len(sentences) * 3))
            weights = [max(1, len(s)) for s in sentences]
            total = sum(weights)
            out: list[dict[str, Any]] = []
            cursor = 0.0
            for i, sentence in enumerate(sentences):
                share = weights[i] / total if total > 0 else 1 / len(sentences)
                seg_dur = max(0.8, duration * share)
                start = cursor
                end = min(duration, cursor + seg_dur)
                cursor = end
                out.append({"text": sentence, "start": start, "end": end})
            return out

        # Pack sentence units into pause-defined speech ranges by duration weight.
        range_durs = [max(0.3, e - s) for s, e in ranges]
        total_range_dur = sum(range_durs)
        total_chars = sum(max(1, len(s)) for s in sentences)
        sentence_idx = 0
        out: list[dict[str, Any]] = []

        for r_i, (r_start, r_end) in enumerate(ranges):
            if sentence_idx >= len(sentences):
                break
            target_chars = max(1, int(total_chars * (range_durs[r_i] / total_range_dur)))
            acc = 0
            picked: list[str] = []
            while sentence_idx < len(sentences):
                s = sentences[sentence_idx]
                picked.append(s)
                acc += max(1, len(s))
                sentence_idx += 1
                if acc >= target_chars and r_i < len(ranges) - 1:
                    break

            text = "".join(picked).strip()
            if text:
                out.append({"text": text, "start": r_start, "end": r_end})

        if sentence_idx < len(sentences):
            tail_text = "".join(sentences[sentence_idx:]).strip()
            if out and tail_text:
                out[-1]["text"] = f"{out[-1]['text']}{tail_text}"
            elif tail_text:
                last_end = ranges[-1][1]
                out.append({"text": tail_text, "start": max(0.0, last_end - 2.0), "end": last_end})
        return out

    @staticmethod
    def _normalize_time_scale(raw_segments: list[dict[str, Any]], duration_seconds: float | None) -> None:
        if not raw_segments or duration_seconds is None or duration_seconds <= 0:
            return

        times: list[float] = []
        for seg in raw_segments:
            for k in ("start", "end"):
                v = seg.get(k)
                if isinstance(v, (int, float)):
                    times.append(float(v))
        if not times:
            return

        max_time = max(times)
        scale = 1.0
        if max_time > duration_seconds * 3:
            for candidate in (1000.0, 100.0, 10.0):
                if (max_time / candidate) <= duration_seconds * 1.5:
                    scale = candidate
                    break

        if scale != 1.0:
            for seg in raw_segments:
                for k in ("start", "end"):
                    v = seg.get(k)
                    if isinstance(v, (int, float)):
                        seg[k] = float(v) / scale

        for seg in raw_segments:
            s = seg.get("start")
            e = seg.get("end")
            if isinstance(s, (int, float)):
                seg["start"] = max(0.0, min(float(s), duration_seconds))
            if isinstance(e, (int, float)):
                seg["end"] = max(0.0, min(float(e), duration_seconds))

    @staticmethod
    def _segment_to_text(segment: dict[str, Any]) -> str:
        for key in ("text", "sentence", "content"):
            value = segment.get(key)
            if value:
                return str(value).strip()
        return ""

    @staticmethod
    def _segment_time(segment: dict[str, Any]) -> tuple[float | None, float | None]:
        def _to_float(value: Any) -> float | None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        start = _to_float(segment.get("start"))
        end = _to_float(segment.get("end"))

        if start is None or end is None:
            ts = segment.get("timestamp") or segment.get("time_stamps")
            if isinstance(ts, (list, tuple)) and len(ts) >= 2:
                start = _to_float(ts[0])
                end = _to_float(ts[1])

        return start, end

    @staticmethod
    def _tokenize_for_display(chinese_text: str) -> list[str]:
        raw_tokens = jieba.lcut(chinese_text)
        tokens: list[str] = []
        for token in raw_tokens:
            token = token.strip()
            if not token:
                continue
            if re.fullmatch(r"[\u3002\uff0c\uff1f\uff01,.!?;:]+", token):
                continue
            tokens.append(token)
        return tokens

    @staticmethod
    def _to_pinyin(word: str) -> str:
        syllables = pinyin(word, style=Style.TONE3, heteronym=False)
        return " ".join(item[0] for item in syllables if item)

    @staticmethod
    def _clean_mlx_text(text: str) -> str:
        cleaned = text.strip().splitlines()[0].strip()
        cleaned = cleaned.strip("\"'")
        return cleaned

    def _translate_full(self, text: str, backend: str, mlx_model_name: str) -> str:
        if backend == "mlx":
            model, tokenizer, generate_fn = self._load_mlx(mlx_model_name)
            prompt = (
                "Translate the following Chinese sentence into natural English.\n"
                "Return only the English translation.\n"
                f"Chinese: {text}\nEnglish:"
            )
            out = generate_fn(model, tokenizer, prompt=prompt, max_tokens=160)
            return self._clean_mlx_text(out)

        translator = self._load_hf_translator()
        return translator(text, max_length=512)[0]["translation_text"]

    def _translate_words(self, words: list[str]) -> list[str]:
        if not words:
            return []

        translator = self._load_hf_translator()
        token_results = translator(words, max_length=64)
        return [item["translation_text"] for item in token_results]

    def _build_segment_output(
        self,
        segment_text: str,
        start: float | None,
        end: float | None,
        translation_backend: str,
        mlx_model_name: str,
    ) -> SegmentOutput:
        tokens = self._tokenize_for_display(segment_text)
        full_english = self._translate_full(segment_text, translation_backend, mlx_model_name) if segment_text else ""
        english_tokens = self._translate_words(tokens)

        word_tokens = [
            WordToken(word=word, pinyin=self._to_pinyin(word), english=eng)
            for word, eng in zip(tokens, english_tokens)
        ]

        return SegmentOutput(
            start=start,
            end=end,
            chinese=segment_text,
            tokens=word_tokens,
            full_english=full_english,
        )

    @staticmethod
    def _emit_progress(progress_cb: ProgressCallback | None, percent: int, message: str) -> None:
        if progress_cb is None:
            return
        progress_cb(max(0, min(100, int(percent))), message)

    def run(
        self,
        youtube_url: str,
        translation_backend: str = "hf",
        asr_model_name: str | None = None,
        mlx_model_name: str | None = None,
        progress_cb: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        selected_asr_model = asr_model_name or os.getenv("CNTRANSCRIBE_ASR_MODEL", "Qwen/Qwen3-ASR-1.7B")
        aligner_name = os.getenv("CNTRANSCRIBE_QWEN_ALIGNER", "Qwen/Qwen3-ForcedAligner-0.6B")
        selected_mlx_model = mlx_model_name or os.getenv(
            "CNTRANSCRIBE_MLX_MODEL",
            "mlx-community/Qwen3-4B-8bit",
        )
        self._logger.info(
            "Pipeline config: ASR=%s aligner=%s translation_backend=%s mlx_model=%s",
            selected_asr_model,
            aligner_name,
            translation_backend,
            selected_mlx_model,
        )

        with tempfile.TemporaryDirectory(prefix="cntranscribe-") as tmpdir:
            started = time.time()
            warnings: list[str] = []
            debug_mode = os.getenv("DEBUG", "").strip() == "1"

            self._emit_progress(progress_cb, 2, "Preparing pipeline")
            workdir = Path(tmpdir)
            audio_path = workdir / "source.wav"

            self._emit_progress(progress_cb, 10, "Downloading YouTube audio")
            self._logger.info("Downloading audio from YouTube...")
            downloaded_audio = self._download_youtube_audio(youtube_url, audio_path)
            self._logger.info("Audio ready: %s", downloaded_audio)
            audio_duration_seconds = self._probe_audio_duration_seconds(downloaded_audio)

            if debug_mode:
                self._emit_progress(progress_cb, 16, "DEBUG=1 active: trimming to first 60 seconds")
                downloaded_audio = self._trim_audio_seconds(downloaded_audio, 60)
                audio_duration_seconds = self._probe_audio_duration_seconds(downloaded_audio) or 60.0
                warnings.append("DEBUG=1 active: processed only first 60 seconds of audio.")
                self._logger.info("DEBUG=1 active; trimmed audio to first 60 seconds: %s", downloaded_audio)

            self._emit_progress(progress_cb, 28, f"Loading ASR model ({selected_asr_model})")
            self._logger.info("Loading Qwen3-ASR model and aligner...")
            model, has_aligner = self._load_asr_model(selected_asr_model, aligner_name)
            if not has_aligner:
                warnings.append("Forced aligner not available for this ASR model; timestamps may be less precise.")

            self._emit_progress(progress_cb, 40, "Running ASR")
            self._logger.info("Running ASR with timestamps...")
            asr_results = model.transcribe(
                audio=str(downloaded_audio),
                language="Chinese",
                return_time_stamps=True,
            )

            if not asr_results:
                raise RuntimeError("ASR returned no results.")

            result_item = asr_results[0]
            raw_segments = self._normalize_qwen_segments(result_item)
            if not raw_segments:
                raise RuntimeError("ASR did not return transcription text.")

            self._normalize_time_scale(raw_segments, audio_duration_seconds)

            # If ASR collapses to a single text block, split by punctuation and detected pauses.
            non_empty_segments = [s for s in raw_segments if self._segment_to_text(s)]
            if len(non_empty_segments) <= 1:
                full_text = self._segment_to_text(non_empty_segments[0]) if non_empty_segments else ""
                sentences = self._split_text_sentences(full_text)
                if len(sentences) > 1:
                    pause_ranges = self._detect_pause_segments(downloaded_audio, audio_duration_seconds)
                    rebuilt = self._segments_from_sentences_and_ranges(
                        sentences,
                        pause_ranges,
                        audio_duration_seconds,
                    )
                    if rebuilt:
                        raw_segments = rebuilt
                        warnings.append("ASR returned one block; transcript was split into smaller pause-based segments.")

            selected_backend = "mlx" if translation_backend == "mlx" else "hf"
            if selected_backend == "mlx":
                self._emit_progress(progress_cb, 70, f"Loading MLX translation model ({selected_mlx_model})")
                try:
                    self._load_mlx(selected_mlx_model)
                except Exception as exc:
                    selected_backend = "hf"
                    warnings.append(f"MLX translation backend unavailable, fell back to HF model: {exc}")
                    self._logger.warning("MLX unavailable, falling back to HF: %s", exc)
            self._logger.info(
                "Resolved translation stack: backend=%s model=%s",
                selected_backend,
                selected_mlx_model if selected_backend == "mlx" else "Helsinki-NLP/opus-mt-zh-en",
            )

            outputs: list[SegmentOutput] = []
            total = max(1, len(raw_segments))
            self._logger.info("Translating %d segment(s) using backend=%s...", len(raw_segments), selected_backend)
            for idx, raw_segment in enumerate(raw_segments, start=1):
                segment_text = self._segment_to_text(raw_segment)
                if not segment_text:
                    continue

                start, end = self._segment_time(raw_segment)
                outputs.append(
                    self._build_segment_output(
                        segment_text,
                        start,
                        end,
                        selected_backend,
                        selected_mlx_model,
                    )
                )

                percent = 78 + int((idx / total) * 20)
                self._emit_progress(progress_cb, percent, f"Translating segment {idx}/{total}")
                self._logger.info("Translated segment %d/%d", len(outputs), len(raw_segments))

            if not outputs:
                raise RuntimeError("No text segments were produced from ASR output.")

            # If aligner did not return usable times, derive approximate ranges for UI syncing.
            if all(segment.start is None or segment.end is None for segment in outputs):
                duration = audio_duration_seconds or float(max(30, len(outputs) * 6))
                total_weight = 0
                weights: list[int] = []
                for seg in outputs:
                    w = max(1, len(seg.chinese.strip()))
                    weights.append(w)
                    total_weight += w

                cursor = 0.0
                for idx, seg in enumerate(outputs):
                    share = weights[idx] / total_weight if total_weight > 0 else 1 / len(outputs)
                    seg_dur = max(1.0, duration * share)
                    seg.start = cursor
                    seg.end = min(duration, cursor + seg_dur)
                    cursor = seg.end
                warnings.append("Timestamps were approximated because ASR did not provide alignment times.")

            self._emit_progress(progress_cb, 100, "Complete")
            self._logger.info("Pipeline complete in %.1fs", time.time() - started)
            return {
                "source_url": youtube_url,
                "translation_backend": selected_backend,
                "translation_model": selected_mlx_model if selected_backend == "mlx" else "Helsinki-NLP/opus-mt-zh-en",
                "asr_backend": selected_asr_model,
                "audio_duration_seconds": audio_duration_seconds,
                "warnings": warnings,
                "segments": [
                    {
                        **asdict(segment),
                        "tokens": [asdict(token) for token in segment.tokens],
                    }
                    for segment in outputs
                ],
            }
