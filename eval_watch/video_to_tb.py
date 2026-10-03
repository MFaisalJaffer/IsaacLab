#!/usr/bin/env python
"""Log an animated GIF into a TensorBoard run dir as an image summary.

Usage: video_to_tb.py <gif_path> <tb_run_dir> <global_step>

Writes the GIF bytes straight into a 'policy/rollout' image summary (TensorBoard
renders animated GIFs in the IMAGES tab). We bypass torch's add_video -> moviepy
path, which silently encodes an empty clip in this env; the GIF is the same
ffmpeg-produced one the :8800 page uses, so it's known-good.
"""
import sys

import imageio.v2 as imageio
from tensorboard.compat.proto.summary_pb2 import Summary
from torch.utils.tensorboard import SummaryWriter


def main() -> int:
    gif_path, tb_dir, step = sys.argv[1], sys.argv[2], int(sys.argv[3])
    # optional 4th arg: TB tag (2026-07-21, dual terrain/flat renders)
    tag = sys.argv[4] if len(sys.argv) > 4 else "policy/rollout"
    with open(gif_path, "rb") as f:
        gif_bytes = f.read()
    frame0 = imageio.get_reader(gif_path).get_data(0)
    h, w = int(frame0.shape[0]), int(frame0.shape[1])
    img = Summary.Image(height=h, width=w, colorspace=3, encoded_image_string=gif_bytes)
    summary = Summary(value=[Summary.Value(tag=tag, image=img)])
    sw = SummaryWriter(tb_dir)
    sw._get_file_writer().add_summary(summary, step)
    sw.flush()
    sw.close()
    print(f"video_to_tb: wrote GIF ({len(gif_bytes)} bytes, {w}x{h}) -> {tb_dir} step={step}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
