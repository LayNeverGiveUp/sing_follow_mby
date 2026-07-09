# Source Audio

Put licensed full-song audio files here.

Supported extensions:

- `.wav`
- `.mp3`
- `.m4a`
- `.aac`
- `.flac`

Expected filenames:

- `mao_buyi_xiaochou.wav`
- `mao_buyi_like_me.wav`
- `mao_buyi_summer.wav`
- `mao_buyi_unstained.wav`
- `mao_buyi_ordinary_day.wav`
- `mao_buyi_meat_and_vegetable.wav`
- `mao_buyi_borrow.wav`
- `mao_buyi_murmur.wav`
- `mao_buyi_no_question.wav`
- `mao_buyi_muma_city.wav`
- `mao_buyi_yichengshanlu.wav`

The importer reads WAV directly. MP3 is decoded by the Python `miniaudio` dependency.
Other encoded formats can use `ffmpeg` when installed.

Chinese song names are also accepted, for example `消愁.mp3` with `消愁.lrc`.
