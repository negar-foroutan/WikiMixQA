import json
import pandas as pd


categories = json.load(open("data/category_taxonomy.json"))

for topic, v in categories.items():
    for subtopic in v.keys():
        df = pd.read_json(f"data/{topic}_{subtopic}_chart.json", lines=True)
        n_images, n_charts = [], []
        for i, row in df.iterrows():
            number_images = len(row["images"])
            number_charts = sum([1 for img in row["images"] if img["chart"] == "yes"])
            n_images.append(number_images)
            n_charts.append(number_charts)
        df["n_images"] = n_images
        df["n_charts"] = n_charts
        df.to_json(
            f"data/{topic}_{subtopic}_chart_stats.json",
            orient="records",
            lines=True,
        )
