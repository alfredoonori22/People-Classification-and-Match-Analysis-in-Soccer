from typing import Tuple

import torch
from torchvision.transforms import Resize, functional, ColorJitter


def get_transform(split, tasks):
    if split == "train" and task == "detection":
        return [RandomHorizontalFlip(), RandomPhotometricDistort(), ToTensor()]
    elif split == "train" and task == "fb_people":
        return [RandomPhotometricDistort(), ToTensor()]
    else:
        return [ToTensor()]


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target, size):
        for t in self.transforms:
            image, target = t(image, target, size)
        return image, target


class ResizeImg:
    def __call__(self, image, target, size):
        # Revert tuple order for Resize()
        size = size[::-1]
        # Resize the image
        resize = Resize(size)
        image = resize(image)

        return image, target


class CheckBoxes:
    def __call__(self, image, target, size):
        w, h = size

        boxes = target['boxes']
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        target['boxes'] = boxes

        return image, target


class RandomHorizontalFlip:
    def __call__(self, image, target, size):
        if torch.rand(1) < 0.5:
            image = functional.hflip(image)
            if target is not None:
                target["boxes"][:, [0, 2]] = size[0] - target["boxes"][:, [2, 0]]
        return image, target


class RandomPhotometricDistort:
    def __init__(self, contrast: Tuple[float] = (0.5, 1.5), saturation: Tuple[float] = (0.5, 1.5),
                 hue: Tuple[float] = (-0.05, 0.05), brightness: Tuple[float] = (0.875, 1.125), p: float = 0.5):
        self._brightness = ColorJitter(brightness=brightness)
        self._contrast = ColorJitter(contrast=contrast)
        self._hue = ColorJitter(hue=hue)
        self._saturation = ColorJitter(saturation=saturation)
        self.p = p

    def __call__(self, image, target, size):
        if isinstance(image, torch.Tensor):
            if image.ndimension() not in {2, 3}:
                raise ValueError(f'image should be 2/3 dimensional. Got {image.ndimension()} dimensions')
            elif image.ndimension() == 2:
                image = image.unsqueeze(0)

        r = torch.rand(6)

        if r[0] < self.p:
            image = self._brightness(image)

        contrast_before = r[1] < 0.5
        if r[1] < self.p:
            if r[2] < self.p:
                image = self._contrast(image)

        if r[3] < self.p:
            image = self._saturation(image)

        if r[4] < self.p:
            image = self._hue(image)

        if not contrast_before:
            if r[5] < self.p:
                image = self._contrast(image)

        return image, target


class ToTensor:
    def __call__(self, image, target, size):
        image = functional.to_tensor(image)
        return image, target
