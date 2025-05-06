from pathlib import Path

import datasets as ds


def load_disk_dataset(dataset_path: Path) -> ds.Dataset:
    dataset = ds.load_from_disk(dataset_path)
    assert isinstance(dataset, ds.Dataset)
    return dataset


def load_repo_dataset(repo_id: str, *, split: str | None = "train") -> ds.Dataset:
    dataset = ds.load_dataset(repo_id, split=split)
    assert isinstance(dataset, ds.Dataset)
    return dataset


def load_dataset(id: str | Path, *, split: str | None = "train"):
    match id:
        case Path():
            dataset = load_disk_dataset(id)
        case _:
            dataset = load_repo_dataset(id, split=split)

    return dataset
