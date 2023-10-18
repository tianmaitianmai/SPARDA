# Python

## About Reproducibility
We have conducted tests and found that for the *same dataset*, using the *same random seed* in the *same environment* on the *same computer*, our neural network always produces the same model and performance results. However, we *cannot* guarantee the same results for different `PyTorch` versions. Slight disruptions in the order of random number generation, such as adjusting the execution sequence of the code, may lead to slightly different models.

See more from PyTorch's Official [Docs](https://pytorch.org/docs/stable/notes/randomness.html): *Completely reproducible results are not guaranteed across PyTorch releases, individual commits, or different platforms. Furthermore, results may not be reproducible between CPU and GPU executions, even when using identical seeds.*

In order to facilitate the replication of our results for anyone interested in trying our code, we have made our previously trained and utilized model parameters available to the public. The link can be found [here](../../data/READEME.md).

## utils.py
Define some utility functions.

## global_var.py
Set some global variables, like `RANDOM_SEED`.
The `RANDOM_SEED` is set to maximize reproducibility as much as possible.

## model.py
Define the model.

## dataloader.py
Define the dataloader and transformations that might be applied on the data.

## train.py
Define the `FocalLoss` Module, the epochs, the paths for loading data, and some other hyperparameters used during training.
They are hard-coded in the source code.

## predict.py
Predict(or Inference) for data in the given paths.
The paths are hard-coded.

## lat_adjustment.py
Due to the transformation of the coordinate system, apply a correction to the latitude.
