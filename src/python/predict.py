import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision.utils import save_image
from functools import reduce
from tqdm.auto import tqdm
import torchvision.transforms.functional as TF
from torch.nn import functional as F
import os

from utils import *
from dataloader import *
from model import *

# Set device type
device = "cuda" if torch.cuda.is_available() else "cpu"

def inference_and_save(model: torch.nn.Module,
                       x1_paths,
                       x2_paths,
                       x3_paths,
                       y_softmax_paths,
                       y_onehot_paths,
                       transform,
                       device: torch.device = device):
    model = model.to(device)
    model.eval()
    assert len(x1_paths) == len(x2_paths)==len(x3_paths)
    with torch.inference_mode():
        for x1_path, x2_path, x3_path, y_softmax_path, y_onehot_path in zip(x1_paths, x2_paths, x3_paths, tqdm(y_softmax_paths), y_onehot_paths):
            with Image.open(x1_path) as x1, Image.open(x2_path) as x2, Image.open(x3_path) as x3:
                x = transform([x1, x2, x3])
                x = torch.unsqueeze(x.to(device), 0)
                y_pred = model(x)[0,:,:,:]
                y_pred = torch.softmax(y_pred, dim=0)
                save_image(y_pred, y_softmax_path)
                y_pred = 1.0*F.one_hot(torch.argmax(y_pred, dim=0), num_classes=3)
                y_pred = y_pred.permute(2,0,1)
                #print(y_pred.shape,"\n")
                save_image(y_pred, y_onehot_path)

if __name__ == "__main__":

    file_dir = "../../data/"
    x1_perfix = "channel_1/"
    x2_perfix = "channel_2/"
    x3_perfix = "channel_3/"
    y_softmax_perfix = file_dir + "predict_softmax/"
    y_onehot_perfix = file_dir + "predict_onehot/"
    mkdir_if_not_exists(y_softmax_perfix)
    mkdir_if_not_exists(y_onehot_perfix)
    x1_paths = list(sorted(Path(file_dir).joinpath(x1_perfix).glob("*/*.png")))
    x2_paths = list(sorted(Path(file_dir).joinpath(x2_perfix).glob("*/*.png")))
    x3_paths = list(sorted(Path(file_dir).joinpath(x3_perfix).glob("*/*.png")))
    y_softmax_paths = [Path(y_softmax_perfix).joinpath(get_n_last_subparts_path(x,1)) for x in x1_paths]
    y_onehot_paths = [Path(y_onehot_perfix).joinpath(get_n_last_subparts_path(x,1)) for x in x1_paths]

    print([len(x) for x in [x1_paths, x2_paths, x3_paths, y_softmax_paths, y_onehot_paths]])

    mkdir_if_not_exists(y_softmax_perfix)
    mkdir_if_not_exists(y_onehot_perfix)

    model = UNet(in_channels=3, out_channels=3, features=[64, 128, 256, 512, 1024])
    model.load_state_dict(torch.load(save_path + "model/m_best.pt"))
    model.to(device);

    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    _ch1_path_ = Path(file_dir).joinpath(x1_perfix)
    sub_dirs =[x for x in os.listdir(_ch1_path_) if os.path.isdir(_ch1_path_.joinpath(x))]

    for x in sub_dirs :
        mkdir_if_not_exists(y_softmax_perfix + str(x) + "/")
        mkdir_if_not_exists(y_onehot_perfix + str(x) + "/")

    inference_and_save(model, x1_paths, x2_paths, x3_paths, y_softmax_paths, y_onehot_paths, resize_x, device)