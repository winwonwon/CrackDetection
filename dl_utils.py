import torchvision.transforms as T

def get_transform(train):
    if train:
        transforms = [
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]
    else:
        transforms = [
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]
    return T.Compose(transforms)

def collate_fn(batch):
    return tuple(zip(*batch))
