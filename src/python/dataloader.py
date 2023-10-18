import torch
import random
import pathlib
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from global_var import *

def my_transform(xs, y, size=(HEIGHT, WIDTH), seed=RANDOM_SEED):

    if seed is not None:
        random.seed(seed)

    resize = transforms.Resize(size=size)
    img_h, img_w = size
    xs = [resize(x) for x in xs]
    y = resize(y)

    # Random horizontal flipping
    if random.random() > 0.5:
        xs = [TF.hflip(x) for x in xs]
        y = TF.hflip(y)

    # Random affine
    affine_param = transforms.RandomAffine.get_params(
        degrees=[-0.5, 0.5],
        translate=[0.005, 0.0],
        img_size=[img_w, img_h],
        scale_ranges=[1, 1],
        shears=[-20, 20],
    )

    affine_trans = lambda z: TF.affine(z, 
                      affine_param[0], affine_param[1],
                      affine_param[2], affine_param[3])

    xs = [affine_trans(x) for x in xs]
    y = affine_trans(y)

    xs = [TF.to_tensor(x) for x in xs]
    y = TF.to_tensor(y)

    # make sure the black([0,0,0]) in tensor y transformed
    # to be blue([0,0,1], the background)
    y_mask = torch.max(y, dim=0)[0] == 0
    y_mask = y_mask.unsqueeze(dim=0)
    zeros = torch.zeros_like(y_mask)
    y_mask = torch.cat([zeros, zeros, y_mask], dim=0)
    y = y.masked_fill(y_mask, value=1.0)

    xs = torch.cat(xs, dim=0)
    return xs, y


def resize_xy(xs, y, size=(HEIGHT, WIDTH)):
    resize = transforms.Resize(size=size)
    xs = [resize(x) for x in xs]
    y = resize(y)
    xs = [TF.to_tensor(x) for x in xs]
    y = TF.to_tensor(y)
    xs = torch.cat(xs, dim=0)
    return xs, y


def resize_x(xs, size=(HEIGHT, WIDTH)):
    resize = transforms.Resize(size=size)
    xs = [resize(x) for x in xs]
    xs = [TF.to_tensor(x) for x in xs]
    xs = torch.cat(xs, dim=0)
    return xs


class SegmentationDataLoader(Dataset):
    def __init__(
        self,
        file_dir: str,
        xs_perfixs=["ch1", "ch2", "ch3"],
        y_perfix="label",
        transform=None,
    ) -> None:
        self.xs_paths = [
            list(sorted(pathlib.Path(file_dir).joinpath(x_perfix)
                        .glob("*.png")))
            for x_perfix in xs_perfixs
        ]
        self.y_paths = list(
            sorted(pathlib.Path(file_dir).joinpath(y_perfix).glob("*.png"))
        )
        self.transform = transform

    def load_pairs(self, index: int) -> Image.Image:
        xs_path = [x[index] for x in self.xs_paths]
        y_path = self.y_paths[index]
        xs = [Image.open(x) for x in xs_path]
        y = Image.open(y_path)
        return xs, y

    def __len__(self) -> int:
        return len(self.y_paths)

    def __getitem__(self, index: int):
        # the returned 'y' is not of the shape (c, h, w)
        # the returned 'y' has the shape (h, w) and each pixel
        # is in range [0,1,2]

        xs, y = self.load_pairs(index)

        if not self.transform:
            xs = torch.cat([TF.to_tensor(x) for x in xs], dim=0)
            y = (TF.to_tensor(y)).argmax(dim=0)
            return xs, y

        xs, y = self.transform(xs, y)
        return xs, y.argmax(dim=0)
