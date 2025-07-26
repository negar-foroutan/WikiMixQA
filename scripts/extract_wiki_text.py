import wikipediaapi
import json
import pandas as pd

wiki_wiki = wikipediaapi.Wikipedia(
    user_agent="MyProjectName (merlin@example.com)",
    language="en",
    extract_format=wikipediaapi.ExtractFormat.WIKI,
)

categories = json.load(open("data/category_taxonomy.json"))

for topic, v in categories.items():
    for subtopic in v.keys():
        df = pd.read_json(f"data/{topic}_{subtopic}_chart_stats.json", lines=True)
        texts = []
        for i, row in df.iterrows():
            print(f"{topic} - {subtopic}: Fetching text for {row['title']}")
            page = wiki_wiki.page(row["title"])
            if page.exists():
                texts.append(page.text)
            else:
                texts.append("")
        df["text"] = texts
        df[["qid", "url", "title", "category", "text"]].to_json(
            f"data/{topic}_{subtopic}_text.json", orient="records", lines=True
        )
