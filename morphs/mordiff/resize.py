import glob
import os

from PIL import Image
from tqdm import tqdm


dir = "/media/blazkowiz47/work/feret"
os.makedirs(dir.replace("feret", "feret/resized/test"), exist_ok=True)
os.makedirs(dir.replace("feret", "feret/resized/train"), exist_ok=True)

files = glob.glob(os.path.join(dir, "**", "*.jpg")) + glob.glob(
    os.path.join(dir, "**", "*.jpeg")
)


for file in tqdm(files):
    img = Image.open(file)
    img = img.resize((256, 256))
    img.save(file.replace("feret", "feret/resized"))
