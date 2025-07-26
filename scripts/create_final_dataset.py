import os
import shutil
import json
import pandas as pd

parent_folder = "final"
categories = json.load(open("data/category_taxonomy.json"))
categories = {"Wikimedia": categories["Wikimedia"]}

for topic, v in categories.items():
    for subtopic in v.keys():
        os.makedirs(os.path.join(parent_folder, topic, subtopic), exist_ok=True)
        df = pd.read_json(f"data/{topic}_{subtopic}_chart_stats.json", lines=True)

        for i, row in df.iterrows():
            qid = row["qid"]

            # copy images
            qid_folder = os.path.join(parent_folder, topic, subtopic, qid)
            os.makedirs(qid_folder, exist_ok=True)
            input_image_folder = f"images/{qid}"
            output_image_folder = os.path.join(qid_folder, "images")
            shutil.copytree(input_image_folder, output_image_folder, dirs_exist_ok=True)

            # copy tables
            input_table_file = os.path.join("tables", qid, "tables.jsonl")
            output_table_file = os.path.join(qid_folder, "tables.jsonl")
            if os.path.exists(input_table_file):
                shutil.copy(input_table_file, output_table_file)

            # copy html
            title = row["title"]
            filename = title.replace(" ", "_")
            filename = filename.replace("/", "_")
            input_html_file = os.path.join(
                f"html/{topic}/{subtopic}", f"{filename}.html"
            )
            output_html_file = os.path.join(qid_folder, f"{filename}.html")
            if os.path.exists(input_html_file):
                shutil.copy(input_html_file, output_html_file)

            # copy generated image from html
            input_html_image_file = os.path.join(
                f"html/{topic}/{subtopic}", f"{filename}.jpg"
            )
            output_html_image_file = os.path.join(qid_folder, f"{filename}.jpg")
            if os.path.exists(input_html_image_file):
                shutil.copy(input_html_image_file, output_html_image_file)

            # metadata
            output_metadata_file = os.path.join(qid_folder, "metadata.json")
            with open(output_metadata_file, "w") as f:
                json.dump(
                    {
                        "qid": qid,
                        "title": title,
                        "url": row["url"],
                        "charts": [
                            i["filename"] for i in row["images"] if i["chart"] == "yes"
                        ],
                    },
                    f,
                )

        df = pd.read_json(f"data/{topic}_{subtopic}_text.json", lines=True)
        for i, row in df.iterrows():
            qid = row["qid"]
            text = row["text"]
            qid_folder = os.path.join(parent_folder, topic, subtopic, qid)
            output_text_file = os.path.join(qid_folder, "wiki.txt")
            with open(output_text_file, "w") as f:
                f.write(text)
