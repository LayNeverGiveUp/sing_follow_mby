# Source Lyrics

Put licensed timestamp lyric files here. `.lrc` is recommended; `.csv` is also supported.

Expected filenames match source audio IDs, for example:

```text
mao_buyi_xiaochou.lrc
```

LRC format:

```text
[00:12.34]first lyric line
[00:16.98]next lyric line
```

For LRC files, each line's end time is inferred from the next line's start time.

Optional CSV format:

```csv
line_id,start_ms,end_ms,text
line_001,12340,16980,first lyric line
line_002,17020,21300,next lyric line
```

`line_id` is optional; if omitted, the importer assigns `line_001`, `line_002`, etc.
