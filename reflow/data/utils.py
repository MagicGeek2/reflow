from torchvision import transforms as T
import torch

def array_to_tensor(array):
    return torch.from_numpy(array)

def get_image_transforms(train=False, random_flip=False):
    image_transforms=[array_to_tensor]
    if train:
        if random_flip:
            image_transforms.append(T.RandomHorizontalFlip())
    else:
        ...
    image_transforms = T.Compose(image_transforms)
    return image_transforms

