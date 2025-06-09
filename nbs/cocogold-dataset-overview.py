from cocogold.dataset import CocoGoldIterableDataset

base_data_dir = "/data/datasets/coco/2017"

# I think there's a bug with parallel loaders,
# so we just iterate sequentially through items
train_dataset = CocoGoldIterableDataset(
    base_data_dir,
    split="train",
    return_type="pt",
)

from collections import defaultdict
from tqdm import tqdm

all_labels = defaultdict(int)
for x in tqdm(train_dataset):
    all_labels[x["class"]] += 1

import matplotlib.pyplot as plt
plt.figure(figsize=(20, 4))
plt.xticks(rotation=90)
plt.plot(all_labels.keys(), all_labels.values())
plt.savefig("coco-training.png")

sorted_labels = {k: v for k, v in sorted(all_labels.items(), key=lambda item: item[1])}

plt.figure(figsize=(10, 16))
bars = plt.barh(sorted_labels.keys(), sorted_labels.values());
plt.bar_label(bars, padding=6);
plt.savefig("coco-training-sorted.png")

