import subprocess, os
import numpy as np

class VideoReader(object):

    def __init__(self, fname, width, height, offset=0):
        self.fname = fname
        self.width = width
        self.height = height
        self.proc = subprocess.Popen([
            'ffmpeg',
            '-ss', str(offset),
            '-i', fname,
            '-f', 'image2pipe',
            '-pix_fmt', 'rgb24',
            '-vf', 'scale=%d:%d' % (height, width),
            '-vcodec', 'rawvideo',
            '-'], stdout=subprocess.PIPE)

    @property
    def frame_bytes(self):
        return self.width * self.height * 3

    def get_frame(self):
        buf = self.proc.stdout.read(self.frame_bytes)
        return np.frombuffer(buf, dtype=np.uint8).reshape((self.width, self.height, 3))

    def close(self):
        self.proc.stdout.close()
