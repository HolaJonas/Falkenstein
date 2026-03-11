from torchvision.transforms import Compose, Resize, RandomHorizontalFlip, RandomRotation
from PIL import Image
from pathlib import Path
import os 

def augment_image(path):
    image_transforms = Compose(
        [Resize(400), RandomHorizontalFlip(p=0.5), RandomRotation(20)]
    )
    orig_img = Image.open(Path(path))
    augmented_img = image_transforms(orig_img)

    new_path = path.split(".")
    new_path[1] = new_path[1] + "a"
    new_path = ".".join(new_path)
    augmented_img.save(Path(new_path))


def augment_dataset(path):
    for subdir, dirs, images in os.walk(Path(path)):
        for image in images:
            cur_path = os.path.join(subdir, image)
            print(cur_path)

