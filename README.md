# cocogold

> Marigold-based, text-grounded CoCo Segmentation

This is just the dataset preparation code. The training code [lives here](https://github.com/pcuenca/Marigold/tree/cocogold).

## Usage

### Installation

Install from the GitHub
[repository](https://github.com/pcuenca/cocogold):

``` sh
$ pip install git+https://github.com/pcuenca/cocogold.git
```

## How to use

Download a copy of the COCO 2017 dataset:

```bash
uv pip install huggingface-hub

huggingface-cli download --local-dir coco-2017 pcuenq/coco-2017-mirror

cd coco-2017
unzip annotations_trainval2017.zip
rm annotations_trainval2017.zip

mkdir images
unzip val2017.zip -d images
rm val2017.zip
ln -s val2017 images/val

unzip train2017.zip -d images
rm train2017.zip
ln -s train2017 images/train
```

Then you can build a PyTorch dataset like this:

```py
from cocogold.dataset import CocoGoldIterableDataset
ds = CocoGoldIterableDataset("coco-2017", split="val", return_type="pt")
```

By default, the dataset will randomly crop 512x512 squares that include a segmentation mask for one of the categories. Each time you iterate through the dataset you'll get different results.

