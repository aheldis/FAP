import torch

IMAGENET_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMAGENET_STD = (0.26862954, 0.26130258, 0.27577711)

mu = torch.tensor(IMAGENET_MEAN).view(3, 1, 1).cuda()
std = torch.tensor(IMAGENET_STD).view(3, 1, 1).cuda()

def normalize(X):
    return (X - mu) / std

def clip_img_preprocessing(X):
    img_size = 224
    X = torch.nn.functional.upsample(X, size=(img_size, img_size), mode='bicubic')
    X = normalize(X)
    return X