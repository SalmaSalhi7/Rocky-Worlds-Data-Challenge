import subprocess
from pathlib import Path

import pandas as pd


TARGET = "lhs1140b"  # use "gj3929b" or "lhs1140b"
ECLIPSE = 6         # change this to the eclipse you want

datasets = {
    "lhs1140b": "stsci/rocky-worlds-lhs-1140b-simulations",
}

manifest_path = Path(f"/home/ssalhi/scratch/ROCKY_WORLDS_DATA/{TARGET}/manifests/manifest.csv")
output_dir = Path(f"/home/ssalhi/scratch/ROCKY_WORLDS_DATA/{TARGET}/LHS_1140b_Simulations/eclipse{ECLIPSE:2d}")
output_dir.mkdir(parents=True, exist_ok=True)

manifest = pd.read_csv(manifest_path)

file_column = "filename"
path_column = "relative_path"
eclipse_column = "eclipse"

eclipse_files = manifest.loc[manifest[eclipse_column] == f"eclipse{ECLIPSE}", [file_column, path_column]]
print(f"Found {len(eclipse_files)} files for eclipse {ECLIPSE} for {TARGET}.")

for _, row in eclipse_files.iterrows():
    print(f"Downloading {row[file_column]}")
    subprocess.run(
        f'kaggle datasets download -d {datasets[TARGET]} -f "{row[path_column]}" -p "{output_dir}" --unzip',
        shell=True,
        check=True,
    )