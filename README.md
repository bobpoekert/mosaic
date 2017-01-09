# mosaic
video patch synthesis

dependencies
---
* ffmpeg
* [nmslib](https://github.com/searchivarius/nmslib)
* numpy
* scikit-image
* PIL/Pillow

usage
---

`python test.py <video file> <output directory> <video start time in seconds>`

This will read five seconds of video from `video file` starting at `start time`, index the patches from that video, and generate frames as png images in `output directory`. If you want, you can then use ffmpeg to turn those frames into a video file.

`test.py` is very short and simple, so you're encouraged to start hacking it. There's also a lot of commented-out experiments in `patches.py`, which you're encouraged to uncomment and see how that changes your results.
