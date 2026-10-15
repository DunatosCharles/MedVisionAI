import argparse
import json
import os

import numpy as np
from PIL import Image

from src.data.busi_dataset import BUSIDataset
from src.data.resnet_dataloader import BUSI_ROOT, BUSI_GROUPS_FILE


def dhash(image_path, size=16):

    # Difference hash: compares neighbouring pixels of a small grayscale
    # thumbnail, so it survives small shifts, crops and brightness changes
    image = Image.open(image_path).convert("L").resize(
        (size + 1, size),
        Image.LANCZOS
    )

    pixels = np.asarray(image, dtype=np.float32)

    return (pixels[:, 1:] > pixels[:, :-1]).flatten()


def find_groups(hashes, threshold):

    distances = (hashes[:, None, :] != hashes[None, :, :]).sum(-1)

    parent = list(range(len(hashes)))

    def find(x):

        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]

        return x

    # Union-find: images connected through near-duplicate pairs share a group
    for a, b in zip(*np.where(np.triu(distances <= threshold, 1))):
        parent[find(a)] = find(b)

    roots = [find(x) for x in range(len(hashes))]

    # Renumber groups 0..n-1 in file order
    ids = {}

    return [ids.setdefault(root, len(ids)) for root in roots]


def main():

    parser = argparse.ArgumentParser(
        description="Group near-duplicate BUSI images by lesion"
    )

    # Visually checked: pairs up to ~60 bits apart are frames of the same
    # lesion, pairs above ~65 are different lesions
    parser.add_argument("--threshold", type=int, default=55)

    args = parser.parse_args()


    dataset = BUSIDataset(root_dir=BUSI_ROOT)

    hashes = np.array([dhash(path) for path in dataset.images])

    group_ids = find_groups(hashes, args.threshold)


    names = [
        os.path.relpath(path, BUSI_ROOT)
        for path in dataset.images
    ]

    members = {}

    for index, group in enumerate(group_ids):
        members.setdefault(group, []).append(index)


    # The same image filed under both benign and malignant: the true
    # label is unknown, so these are left out of training and evaluation
    conflicting = sorted(
        names[i]
        for indices in members.values()
        if len({dataset.labels[i] for i in indices}) > 1
        for i in indices
    )


    with open(BUSI_GROUPS_FILE, "w") as f:
        json.dump(
            {
                "threshold": args.threshold,
                "groups": dict(zip(names, group_ids)),
                "conflicting_labels": conflicting
            },
            f,
            indent=2
        )


    multi = [m for m in members.values() if len(m) > 1]

    print(f"Images: {len(names)}")
    print(f"Lesion groups: {len(members)}")
    print(
        f"Groups with near-duplicates: {len(multi)} "
        f"({sum(len(m) for m in multi)} images)"
    )
    print(f"Images with conflicting labels: {len(conflicting)}")
    print(f"Saved {BUSI_GROUPS_FILE}")


if __name__ == "__main__":

    main()
