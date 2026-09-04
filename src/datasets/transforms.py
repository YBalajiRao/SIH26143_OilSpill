import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

class SpeckleNoise(A.ImageOnlyTransform):
    def __init__(self, variance=0.05, p=0.5):
        super().__init__(p=p)
        self.variance = variance
    def apply(self, img, **params):
        img_f = img.astype(np.float32)
        noise = np.random.normal(1.0, np.sqrt(self.variance), img_f.shape).astype(np.float32)
        return np.clip(img_f * noise, 0.0, 1.0)

def get_train_transforms(image_size=(256, 256), speckle_var=0.05):
    return A.Compose([
        A.Resize(image_size[0], image_size[1]),
        A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5),
        SpeckleNoise(variance=speckle_var, p=0.3),
        ToTensorV2(),
    ])

def get_val_transforms(image_size=(256, 256)):
    return A.Compose([A.Resize(image_size[0], image_size[1]), ToTensorV2()])
