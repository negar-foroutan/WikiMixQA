import requests
import argparse
import pandas as pd
import os


def get_html(title: str, width: int = 400):
    response = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "parse",
            "format": "json",
            "page": title,
            "prop": "text",
        },
    ).json()
    html = response["parse"]["text"]["*"]
    return html


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
        output_file = os.path.join(args.output_dir, f"{filename}.html")
        if os.path.exists(output_file):
            print(f"Skipping {filename}.html")
            continue
        try:
            html = get_html(title)
            with open(output_file, "w") as f:
                f.write(html)
            print(f"Saved {filename}.html")
        except:
            print(f"Could not get {filename}.html")
