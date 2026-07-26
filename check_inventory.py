#!/usr/bin/env python3
import pandas as pd

# Load the inventory data
names = ["_", "theme0", "theme1", "theme2", "theme3", "theme4", "age", "area"]
inventory = pd.read_csv("examples/data/woodstock_model_files_tsa24_clipped/tsa24_clipped.are",
                        delimiter=" ", header=None, names=names)
inventory.drop("_", axis=1, inplace=True)

print("Inventory dtype keys:")
for _, row in inventory.iterrows():
    dtype_key = tuple(str(row["theme%i" % i]) for i in range(5))
    print(f"  {dtype_key}")

print("\nUnique species codes (theme2):")
print(inventory["theme2"].unique())