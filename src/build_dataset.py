from data_loader import load_all
from feature_engineering import add_features
from merge_data import merge_cl2_dxy

def main():
    cl2, dxy = load_all()
    cl2 = add_features(cl2)
    merged = merge_cl2_dxy(cl2, dxy)
    merged.to_csv("data/final_dataset.csv", index=False)
    print("Dataset built → data/final_dataset.csv")

if __name__ == "__main__":
    main()