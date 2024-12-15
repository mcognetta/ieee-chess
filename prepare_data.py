import pandas as pd
import urllib.request
from pathlib import Path
import zstandard as zstd


if __name__ == "__main__":
    if not Path("raw_datasets/lichess_db_puzzle.csv").is_file():
        urllib.request.urlretrieve(
            "https://database.lichess.org/lichess_db_puzzle.csv.zst",
            "raw_datasets/lichess_db_puzzle.csv.zst",
        )
        with open("raw_datasets/lichess_db_puzzle.csv.zst", "rb") as compressed_file:
            dctx = zstd.ZstdDecompressor()
            
            # Read and decompress the data
            with open("raw_datasets/lichess_db_puzzle.csv", "wb") as output_file:
                dctx.copy_stream(compressed_file, output_file)

    for CLEANED in [True, False]:
        df = pd.read_csv("raw_datasets/lichess_db_puzzle.csv")

        # drop rows with RatingDeviation over 90
        if CLEANED:
            df = df[df["RatingDeviation"] <= 90]

        # shuffle the dataframe
        df = df.sample(frac=1, random_state=123).reset_index(drop=True)

        validation_size = 10000
        test_size = 10000

        validation_df = df[:validation_size]
        test_df = df[validation_size : validation_size + test_size]
        train_df = df[validation_size + test_size :]

        if CLEANED:
            train_df.to_csv("datasets/cleaned_train_set.csv", index=False)
            validation_df.to_csv("datasets/cleaned_validation_set.csv", index=False)
            test_df.to_csv("datasets/cleaned_test_set.csv", index=False)

        else:
            train_df.to_csv("datasets/train_set.csv", index=False)
            validation_df.to_csv("datasets/validation_set.csv", index=False)
            test_df.to_csv("datasets/test_set.csv", index=False)
