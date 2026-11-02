
import pandas as pd
from pandas.compat.pickle_compat import Unpickler


class U(Unpickler):
    """Handle old pandas module paths in the 2019 dataset pickle."""

    def find_class(self, mod, name):
        mod = mod.replace(
            "pandas.indexes.",
            "pandas.core.indexes."
        )
        return super().find_class(mod, name)


def load_wafer_dataset(file_path):
    """Load the RevD2 Kaggle LSWMD.pkl dataset."""

    with open(file_path, "rb") as f:
        unpickler = U(f)
        unpickler.encoding = "latin1"
        df = unpickler.load()

    return df


if __name__ == "__main__":
    dataset_path = "data/raw/LSWMD.pkl"

    print("Loading WM-811K dataset...")

    df = load_wafer_dataset(dataset_path)

    print("\nDataset loaded successfully!")
    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nFirst 3 rows:")
    print(df.head(3))
