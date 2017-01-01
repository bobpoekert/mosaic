import patches
from PIL import Image

index = patches.PatchIndex()
index.load_video('infomercial.mkv')
index.init_indexes()
outp = index.generate_image(600, 800)
Image.fromarray(outp).save('test.png')
