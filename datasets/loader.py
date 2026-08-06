from pathlib import Path


class DatasetLoader:
    def __init__(self, dataset_root="datasets"):
        self.dataset_root = Path(dataset_root)

        self.datasets = {
            "NEU-DET": self.dataset_root / "NEU-DET",
            "SDNET2018": self.dataset_root / "SDNET2018"
        }

    def verify_datasets(self):
        print("=" * 50)
        print("Dataset Verification")
        print("=" * 50)

        for name, path in self.datasets.items():
            if path.exists():
                num_files = sum(1 for _ in path.rglob("*") if _.is_file())
                print(f"{name:<12} : Found")
                print(f"Location     : {path}")
                print(f"Total Files  : {num_files}\n")
            else:
                print(f"{name:<12} : Not Found\n")


if __name__ == "__main__":
    loader = DatasetLoader()
    loader.verify_datasets()