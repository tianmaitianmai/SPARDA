# Data

## Archive
We list the data that we have decided to make public below. Since GitHub is not suitable for storing large files, we have stored the files in the cloud drive and provided a link below. To ensure that the data is not tampered with, we provide the corresponding sha256sum as verification below.

| Archive | link | sha256sum |
| -------- | ------- | ------ |
| m_best.pt | [download](https://drive.google.com/file/d/1RaL8qb2aVMh901EgowXeIm3Y3RMU30Hy/view?usp=share_link) | eb702d76e960c65f6eedabfba1c37cead528621d74da4802cce200fad1d0ac1b |
| predict_onehot.zip | [download](https://drive.google.com/file/d/1KfP1NZgx9qq_1deKfvQqWZPPB_frCp4T/view?usp=share_link) | 5d4a761d7d2d2938d865a58d923a9466fb0d3c690d5402ac883c3603ae4ddf92 |
| predict_post.zip | [download](https://drive.google.com/file/d/1ADiUdLH3lQHk0Z6KtywiS6dKtOKFvxyw/view?usp=share_link) | be2c5dc6c23176a30a183544a60963ac61a90c2fa0719378562aa0b831e98964 |
| statistic.zip | [download](https://drive.google.com/file/d/1Uoc6WzqXG0F4tRFE3mb05IRlqjBIXBPc/view?usp=share_link) | 9c0c6029d2d2905bc9173db44f357c7d97fad7f7bcb22cae4b34de24210a3f03 |
| test.zip | [download](https://drive.google.com/file/d/1zJRKS-HcL_tNyi4L0Sp0ycw4Km5cgJpy/view?usp=share_link) | 743e99f7967b00882af5c6509e0ea1a7dcbca811bd3aa7a091c70b15c6a643e1 |
| train.zip | [download](https://drive.google.com/file/d/1XkD4m3TLqr1oIIk6n7y0J-gjnoC0l4xA/view?usp=share_link) | 8fb32531df32b6f9176e307a08ac0d2a0431ea7f468fd1e8d4636c9b2e328927 |
| validate.zip | [download](https://drive.google.com/file/d/10fqOSGbKLkHpAAKWNc_XhFI6_cNDLiTL/view?usp=share_link) | 1c013d94e06baa70788acf2f1f54e8ae0779f3a32191b9d5c6784980c66146fb |

## File tree

For convenience during coding, we have established some conventions for the directory structure. Below is an example of a proper directory structure.

```bash
data
├── channel_1 # contain one or more subfolders
│   ├── 2010 # contain `.png` files
│   │   ...
│   └── 2020
├── channel_2
│   ├── 2010
│   │   ...
│   └── 2020
├── channel_3
│   ├── 2010
│   │   ...
│   └── 2020
├── FITS
│   ├── 2010
│   │   ...
│   └── 2020
├── predict_onehot
│   ├── 2010
│   │   ...
│   └── 2020
├── predict_post
│   ├── 2010
│   │   ...
│   └── 2020
├── predict_softmax
│   ├── 2010
│   │   ...
│   └── 2020
└── unet_data
    ├── metric  # contain the metrics recorded during training
    ├── model   # contain the models saved during training
    ├── plot    # contain the plots drawed during training
    ├── test    # test set
    │   ├── ch1 # conatin `.png` files
    │   ├── ch2
    │   ├── ch3
    │   └── label
    ├── train # training set
    │   ├── ch1
    │   ├── ch2
    │   ├── ch3
    │   └── label
    └── validate # validation set
        ├── ch1
        ├── ch2
        ├── ch3
        └── label
```