# Stage 3 Video Input

This stage adds OpenCV-backed `video_file` input while keeping the synthetic
mock source available.

Supported source types:

```text
synthetic
video_file
```

For `video_file`, one `input.path` is opened once per stream. This allows one
local mp4 to simulate 4, 8, or 12 camera streams for later scheduling and
dynamic batch work.

Current scheduling is round-robin:

```text
stream 0 frame 0
stream 1 frame 0
stream 2 frame 0
stream 3 frame 0
stream 0 frame 1
...
```

The current implementation is still single-threaded. Per-stream threaded queues
are a later stage.

No benchmark or FPS conclusions should be made from this stage. The logged
read FPS is only a local runtime observation and must be validated by the user.
