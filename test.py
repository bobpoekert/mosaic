import patches
from PIL import Image
import sys

index = patches.PatchIndex()
index.load_video(sys.argv[1], offset=int(sys.argv[3]))
index.init_indexes()
for i, frame in enumerate(index.generate_video(480, 640, 240)):
    print i
    Image.fromarray(frame).save('%s/%d.png' % (sys.argv[2], i))
