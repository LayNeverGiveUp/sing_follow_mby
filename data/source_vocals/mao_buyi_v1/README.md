# Mao Buyi Separated Vocals

Put licensed vocal-isolated tracks here when the source song audio contains accompaniment.

Accepted names match `data/source_audio/mao_buyi_v1/`:

```text
mao_buyi_xiaochou.wav
消愁.wav
```

`tools/build_catalog.py` uses these files for matching features when present, while prompt clips are still cut from the original source audio.

You can also ask the builder to create this directory with Demucs:

```bash
python tools/build_catalog.py --separation-mode demucs
```

Demucs is optional and must be installed separately.
