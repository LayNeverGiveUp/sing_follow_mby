# Song Followup API

Low-latency demo service for recognizing a short sung phrase and returning a pre-generated follow-up audio clip.

## What This Demo Does

- Exposes a FastAPI WebSocket endpoint for streaming audio-like chunks.
- Maintains a small manifest-driven song catalog.
- Updates candidate matches while chunks arrive.
- Runs a lightweight DTW-style refinement after the client sends `end`.
- Returns the matched song, lyric line, confidence, reply audio URL, and latency metrics.

The current matcher is intentionally deterministic and demo-friendly. It supports JSON `demo_features` messages so the end-to-end service can be tested without a real feature extraction pipeline. Binary chunks are accepted and converted into coarse energy features as a placeholder.

For the Mao Buyi demo path, matching now uses a two-stage flow:

1. ASR text recall narrows lyric candidates when a transcript is available.
2. DTW reranks those candidates with audio features. Without ASR text, DTW falls back to the full catalog.

Audio reranking uses stabilized pitch contour plus interval contour, not raw pitch alone.

## Setup

```bash
cd /Users/lei/Desktop/song-followup-api
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Browser demo:

```text
http://127.0.0.1:8000/demo/
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Demo WebSocket client:

```bash
python scripts/ws_demo_client.py
```

Mao Buyi catalog demo:

```bash
python scripts/ws_mao_demo_client.py
MAO_DEMO_SONG=不染 python scripts/ws_mao_demo_client.py
python scripts/ws_pcm_demo_client.py
```

Optional backend ASR with Volcengine:

```bash
export VOLCENGINE_ASR_API_KEY="..."
export VOLCENGINE_ASR_PUBLIC_BASE_URL="https://your-public-host.example.com"
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The backend uses Volcengine's AUC bigmodel HTTP endpoint. This API accepts a public audio URL, so local browser recordings are saved under `/static/asr/` and submitted as `${VOLCENGINE_ASR_PUBLIC_BASE_URL}/static/asr/<file>.wav`. Do not commit secrets; keep them in your shell environment. `VOLCENGINE_ASR_RESOURCE_ID` defaults to `volc.seedasr.auc`.

Optional local fallback ASR:

```bash
SONG_FOLLOWUP_ASR_MODEL=/path/to/local/faster-whisper-model uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

If backend ASR is unset, the browser demo can still send client-side Web Speech transcripts when the browser supports it. The service also accepts WebSocket transcript messages:

```json
{"type": "asr_transcript", "text": "背上所有的梦与想"}
```

## WebSocket Protocol

Endpoint:

```text
ws://127.0.0.1:8000/v1/realtime-match
```

Initial config:

```json
{
  "type": "start",
  "catalog_id": "kids_songs_v1",
  "sample_rate": 16000,
  "format": "pcm_s16le"
}
```

Streaming messages:

- Binary audio chunks, or
- Demo feature messages:

```json
{
  "type": "demo_features",
  "values": [60, 62, 64, 60]
}
```

End message:

```json
{"type": "end"}
```

Result:

```json
{
  "type": "result",
  "matched": true,
  "song_id": "twinkle_twinkle",
  "song_name": "Twinkle Twinkle Little Star",
  "confidence": 0.98,
  "matched_line": "Twinkle twinkle little star",
  "handoff_type": "next_line",
  "reply_audio_url": "http://127.0.0.1:8000/static/replies/twinkle_next.wav",
  "latency_ms": {
    "stream_duration": 1200,
    "end_to_result": 18,
    "total_session": 1218
  }
}
```

## Replace Demo Matcher With Real Audio Matching

Keep the API contract stable and replace internals in this order:

1. Replace `coarse_pcm_features` with streaming pitch/chroma extraction.
2. Precompute each catalog segment's pitch/chroma features offline.
3. Keep Top-K recall cheap while streaming.
4. Run DTW only on Top-K at `end`.
5. Keep ASR outside the blocking path.

## Catalogs

- `kids_songs_v1`: built-in toy demo catalog.
- `mao_buyi_v1`: Mao Buyi demo catalog using song-title metadata and placeholder melody contours.

The Mao Buyi catalog intentionally does not include copyrighted lyrics or original audio. Replace the placeholder `features` and reply WAV files with licensed audio-derived pitch/chroma features and pre-generated toy-voice clips before accuracy testing.

## Licensed Mao Buyi Assets

The browser demo supports licensed original prompt clips. Put short WAV clips here:

```text
data/prompts/mao_buyi_v1/
```

For example:

```text
data/prompts/mao_buyi_v1/mao_buyi_xiaochou_prompt.wav
```

When you click `随机播放一句`, the UI first tries to play the licensed original clip from `/static/prompts/mao_buyi_v1/`. If the file is missing, it falls back to a synthetic melody prompt.

Put licensed lyric/timestamp files here:

```text
data/lyrics/mao_buyi_v1/
```

## Build A Real Matching Catalog

Put licensed full-song audio files here:

```text
data/source_audio/mao_buyi_v1/
```

If those files contain accompaniment, put separated vocal tracks here:

```text
data/source_vocals/mao_buyi_v1/
```

Accepted filenames match song IDs or Chinese song names. When separated vocals are present, the catalog builder extracts matching features from vocals but still cuts browser prompt clips from the original source audio.

The builder can also call Demucs if it is installed separately:

```bash
python tools/build_catalog.py --separation-mode demucs
```

Supported extensions:

```text
.wav .mp3 .m4a .aac .flac
```

WAV is read directly. MP3/M4A/AAC/FLAC require `ffmpeg` to be installed.
The project also supports MP3 decoding through the Python `miniaudio` dependency in `requirements.txt`.

Both song IDs and Chinese song names are accepted:

```text
data/source_audio/mao_buyi_v1/mao_buyi_xiaochou.mp3
data/source_lyrics/mao_buyi_v1/mao_buyi_xiaochou.lrc

data/source_audio/mao_buyi_v1/消愁.mp3
data/source_lyrics/mao_buyi_v1/消愁.lrc
```

Put timestamped lyric files here:

```text
data/source_lyrics/mao_buyi_v1/
```

LRC format is supported and recommended:

```text
[00:12.34]first lyric line
[00:16.98]next lyric line
```

For LRC files, each line's end time is inferred from the next line's start time.

CSV is also supported:

```csv
line_id,start_ms,end_ms,text
line_001,12340,16980,first lyric line
line_002,17020,21300,next lyric line
```

Build catalog, prompt clips, and reference lyric files:

```bash
python tools/build_catalog.py
```

When an MP3 contains embedded LRC-style lyrics in ID3 tags, the builder uses those embedded lyrics first. External `.lrc` or `.csv` files are used only as fallback.

The builder writes:

- `data/catalog/mao_buyi_v1.json`
- `data/prompts/mao_buyi_v1/*_prompt.wav`
- `data/lyrics/mao_buyi_v1/*.txt`

Restart Uvicorn after rebuilding the catalog so the service reloads the new manifest.
