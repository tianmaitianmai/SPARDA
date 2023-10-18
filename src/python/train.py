import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from functools import reduce
from tqdm.auto import tqdm
import torchvision.transforms.functional as TF
from torch.nn import functional as F
from torch.cuda.amp import GradScaler
from torch import autocast

from global_var import *
from dataloader import *
from model import *
from utils import *

# Set device type
device = "cuda" if torch.cuda.is_available() else "cpu"

class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha: Optional[Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.nll_loss = nn.NLLLoss(weight=alpha, reduction="none")

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        log_p = F.log_softmax(x, dim=1)
        p = torch.exp(log_p)
        ce = self.nll_loss(log_p, y)
        loss = ((1 - p) ** self.gamma) * ce
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / reduce(lambda i,j : i*j, y_true.size()))
    return acc

def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               scaler = None,
               device: torch.device = device):
    train_loss, train_acc = 0.0, 0.0
    for batch, (X, y) in enumerate(tqdm(data_loader)):
        # Send data to GPU
        X, y = X.to(device), y.to(device)

        # 1. Optimizer zero grad
        optimizer.zero_grad()

        if device == 'cuda' and scaler != None:
            # Runs the forward pass with autocasting.
            with autocast(device_type='cuda', dtype=torch.float16):
                # 2. Forward pass
                y_pred = model(X)

                # 3. Calculate loss
                loss = loss_fn(y_pred, y)

            # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
            # Backward passes under autocast are not recommended.
            # Backward ops run in the same dtype autocast chose for corresponding forward ops.
            scaler.scale(loss).backward()

            # scaler.step() first unscales the gradients of the optimizer's assigned params.
            # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
            # otherwise, optimizer.step() is skipped.
            scaler.step(optimizer)

            # Updates the scale for next iteration.
            scaler.update()
        else:
            # 2. Forward pass
            y_pred = model(X)

            # 3. Calculate loss
            loss = loss_fn(y_pred, y)


            # 4. Loss backward
            loss.backward()

            # 5. Optimizer step
            optimizer.step()

        train_loss += loss
        train_acc += accuracy_fn(y_true=y,
                                 y_pred=y_pred.argmax(dim=1)) # Go from logits -> pred labels

    # Calculate loss and accuracy per epoch and print out what's happening
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {(100 * train_acc):.2f}%\n")
    return train_loss, train_acc

def validate_step(data_loader: torch.utils.data.DataLoader,
              model: torch.nn.Module,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device = device):
    validate_loss, validate_acc = 0.0, 0.0
    model.eval() # put model in eval mode
    # Turn on inference context manager
    with torch.inference_mode():
        for X, y in tqdm(data_loader):
            # Send data to GPU
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            y_pred = model(X)

            # 2. Calculate loss and accuracy
            validate_loss += loss_fn(y_pred, y)
            validate_acc += accuracy_fn(y_true=y,
                y_pred=y_pred.argmax(dim=1) # Go from logits -> pred labels
            )

        # Adjust metrics and print out
        validate_loss /= len(data_loader)
        validate_acc /= len(data_loader)
        print(f"Validate loss: {validate_loss:.5f} | Validate accuracy: {(100 * validate_acc):.2f}%\n")
        return validate_loss, validate_acc

def plot_step(data_loader: torch.utils.data.DataLoader,
              model: torch.nn.Module,
              epoch,
              save_path,
              device: torch.device = device,
              n=10):
    model.eval()
    with torch.inference_mode():
        for i, (X, y) in enumerate(data_loader):
            if i >= n:
                break
            # Send data to GPU
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            y_pred = model(X)

            y_pred = nn.Softmax(dim=1)(y_pred)
            X, y, y_pred = X.to("cpu"), y.to("cpu"), y_pred.to("cpu")
            x1 = X[:,0,:,:]
            x2 = X[:,1,:,:]
            x3 = X[:,2,:,:]
            bs = (X.shape)[0]
            for j in range(bs):
                plt.figure()
                fig, ax = plt.subplots(X.shape[1]+3, 1, figsize=(18,8))
                ax[0].imshow(x1[j,:,:], cmap="gray")
                ax[0].axis("off")
                ax[1].imshow(x2[j,:,:], cmap="gray")
                ax[1].axis("off")
                ax[2].imshow(x3[j,:,:], cmap="gray")
                ax[2].axis("off")
                y_j = 1.0*F.one_hot(y[j,:,:], num_classes=3)
                ax[3].imshow(y_j)
                ax[3].axis("off")
                y_j_pred = y_pred[j,:,:,:]
                # save_image(y_j_pred, save_path + "epoch_" + str(epoch) + "_" + str(i) + "_" + str(j) + ".png")
                y_j_pred = y_j_pred.permute(1,2,0)
                ax[4].imshow(y_j_pred)
                ax[4].axis("off")
                y_j_pred_onehot = 1.0*F.one_hot(torch.argmax(y_j_pred,dim=2), num_classes=3)
                ax[5].imshow(y_j_pred_onehot)
                ax[5].axis("off")
                plt.savefig(save_path + "epoch_" + str(epoch) + "_" + str(i) + "_" + str(j) + ".png")
                plt.close()

def main(epochs=100):
    print(device, torch.__version__)
    # Set random seeds
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True

    RNGen = torch.Generator()
    RNGen.manual_seed(RANDOM_SEED)

    train_DS = SegmentationDataLoader(base_path + "train/", transform=my_transform)
    train_dataloader = DataLoader(dataset=train_DS,
                                         batch_size=1,
                                         num_workers=0,
                                         worker_init_fn=seed_worker,
                                         generator=RNGen,
                                         shuffle=True)
    validate_DS = SegmentationDataLoader(base_path + "validate/", transform=resize_xy)
    validate_dataloader = DataLoader(dataset=validate_DS,
                                         batch_size=1,
                                         num_workers=0,
                                         worker_init_fn=seed_worker,
                                         generator=RNGen,
                                         shuffle=False)
    plot_DS = SegmentationDataLoader(base_path + "validate/", transform=resize_xy)
    plot_dataloader = DataLoader(dataset=plot_DS,
                                         batch_size=1,
                                         num_workers=0,
                                         worker_init_fn=seed_worker,
                                         generator=RNGen,
                                         shuffle=False)

    model = UNet(in_channels=3, out_channels=3, features=[64, 128, 256, 512, 1024])
    model = model.to(device)
    #model = torch.compile(model)

    weight = (torch.tensor([2.0, 4.5, 1.0])).to(device)
    loss_fn = FocalLoss(alpha=weight, gamma=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)

    train_losses = np.zeros(epochs)
    train_accs = np.zeros(epochs)

    validate_losses = np.zeros(epochs)
    validate_accs = np.zeros(epochs)

    mkdir_if_not_exists(save_path + "plot/")
    mkdir_if_not_exists(save_path + "model/")

    plot_step(data_loader=plot_dataloader, epoch=0, save_path = save_path + "plot/", model=model, n=1)

    best_validate_loss = 1e9

    for epoch in tqdm(range(epochs)):
        print(f"Epoch: {epoch}\n---------")
        scaler = GradScaler()
        train_loss, train_acc = train_step(data_loader=train_dataloader,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            accuracy_fn=accuracy_fn,
            scaler = scaler
        )
        validate_loss, validate_acc = validate_step(data_loader=validate_dataloader,
            model=model,
            loss_fn=loss_fn,
            accuracy_fn=accuracy_fn
        )

        if validate_loss < best_validate_loss:
            torch.save(model.state_dict(), save_path + "model/m_best.pt")
            best_validate_loss = validate_loss

        train_losses[epoch] = train_loss
        train_accs[epoch] = train_acc

        validate_losses[epoch] = validate_loss
        validate_accs[epoch] = validate_acc

        if (epoch + 1) % 10 == 0:
            plot_step(data_loader=plot_dataloader, model=model, epoch=epoch+1, save_path = save_path + "plot/", n=1)

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), save_path + "model/m_" +  str(epoch + 1) + ".pt")

    #train_losses, train_accs
    mkdir_if_not_exists(save_path + "metric/")
    np.savetxt(save_path + "metric/train_losses.dat", train_losses)
    np.savetxt(save_path + "metric/train_accs.dat", train_accs)
    np.savetxt(save_path + "metric/validate_losses.dat", validate_losses)
    np.savetxt(save_path + "metric/validate_accs.dat", validate_accs)

if __name__ == "__main__":
    main(100)
