import os
import json
import argparse
import pandas as pd

if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--input-dir", type=str, required=True)
    argparser.add_argument("--output-dir", type=str, required=True)
    argparser.add_argument("--suffix", type=str, default="3+")
    args = argparser.parse_args()

    # load category taxonomy
    with open(os.path.join(args.input_dir, "category_taxonomy.json"), "r") as f:
        taxonomy = json.load(f)

    # load categories
    category = pd.read_json(
        os.path.join(args.input_dir, f"category_{args.suffix}.json"), lines=True
    )

    # load infographics
    articles = pd.read_json(
        os.path.join(args.input_dir, f"infographics_{args.suffix}.json"), lines=True
    )
    # merge category and infographics
    df = pd.merge(articles, category, on="qid")
    # explode category column
    df_exploded = df.explode("category")

    for k, v in taxonomy.items():
        for name, category in v.items():
            appended_data = []
            patterns = [cat.replace("*", "") for cat in category if "*" in cat]
            patterns = "|".join(patterns)
            if patterns:
                appended_data.append(
                    df_exploded[df_exploded["category"].str.contains(patterns)]
                )

            for cat in category:
                if "*" not in cat:
                    appended_data.append(df_exploded[df_exploded["category"] == cat])
            matches = pd.concat(appended_data)
            print(f"{k}_{name}: {len(matches)}")
            selection_df = df[df["qid"].isin(matches["qid"].unique())]
            output_filename = args.output_dir + f"/{k}_{name}.json"
            selection_df.to_json(output_filename, orient="records", lines=True)
