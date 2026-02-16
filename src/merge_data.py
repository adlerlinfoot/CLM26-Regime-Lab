def merge_cl2_dxy(cl2, dxy):
    merged = cl2.merge(dxy[["date", "close"]].rename(columns={"close": "dxy"}),
                       on="date",
                       how="inner")
    return merged