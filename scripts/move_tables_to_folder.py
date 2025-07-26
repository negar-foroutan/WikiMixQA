import os
import json
import pandas as pd

filename = "/mnt/tamedia/wikitables/en.jsonl"
parent_folder = "tables"
categories = json.load(open("data/category_taxonomy.json"))
# categories = {"Wikimedia": categories["Wikimedia"]}

qids = []
for topic, v in categories.items():
    for subtopic in v.keys():
        df = pd.read_json(f"data/{topic}_{subtopic}.json", lines=True)
        qids.extend(df["qid"])
d_qids = {i: [] for i in qids}
print(f"Total qids: {len(d_qids)}")


with open(filename) as f:
    for line in f:
        data = json.loads(line)
        if data["wikidata"] not in d_qids:
            continue
        d_qids[data["wikidata"]].append(data)

for qid, data in d_qids.items():
    output_folder = os.path.join(parent_folder, qid)
    os.makedirs(output_folder, exist_ok=True)
    json_data = pd.DataFrame(data).to_json(
        os.path.join(output_folder, "tables.jsonl"), orient="records", lines=True
    )
