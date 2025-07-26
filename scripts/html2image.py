import imgkit
import argparse
import pandas as pd
import os

if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--input-file", type=str, required=True)
    argparser.add_argument("--output-dir", type=str, required=True)
    args = argparser.parse_args()

    df = pd.read_json(args.input_file, lines=True)
    print(df.head())
    print(df.info())

    for i, row in df.iterrows():
        title = row["title"]
        filename = title.replace(" ", "_")
        filename = filename.replace("/", "_")
        output_file = os.path.join(args.output_dir, f"{filename}.jpg")
        if os.path.exists(output_file):
            print(f"Skipping {filename}.jpg")
            continue
        try:
            imgkit.from_url(row["url"], output_file)
            print(f"Saved {filename}.jpg")
        except:
            print(f"Could not get {filename}.jpg")
