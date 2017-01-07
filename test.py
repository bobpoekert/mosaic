import patches
from PIL import Image

index = patches.PatchIndex()
index.load_video('infomercial.mkv')
index.init_indexes()
for i, frame in enumerate(index.generate_video(480, 640, 240)):
    print i
    Image.fromarray(frame).save('frames/%d.png' % i)
