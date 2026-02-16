import pandas as pd
from data_downloader import download_raw_data
from data_loader import load_all
from feature_engineering import add_features
from merge_data import merge_cl2_dxy

def main():
    print("Downloading raw data...")
    download_raw_data()

    print("Loading series...")
    cl2, dxy = load_all()

    print("Building features...")
    cl2 = add_features(cl2)

    print("Merging...")
    df = merge_cl2_dxy(cl2, dxy)
    df = df.dropna().reset_index(drop=True)

    print("Saving final_dataset.csv...")
    df.to_csv("data/final_dataset.csv", index=False)

    print("Done.")

if __name__ == "__main__":
    main()