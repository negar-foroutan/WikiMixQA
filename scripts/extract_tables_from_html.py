import os
import json
import sys
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from core.parse_wikitable_html import extract_html_tables_from_html

categories = json.load(open("data/category_taxonomy.json"))


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--input-dir", type=str, required=True)
    args = argparser.parse_args()

    for topic, v in categories.items():
        topic_dir = os.path.join(args.input_dir, topic)
        for subtopic in v.keys():
            subtopic_dir = os.path.join(topic_dir, subtopic)
            for qid in os.listdir(subtopic_dir):
                if not os.path.isdir(os.path.join(subtopic_dir, qid)):
                    continue
                output_filename = os.path.join(subtopic_dir, qid, "html_tables.json")
                if os.path.exists(output_filename):
                    print(f"Skipping {qid}")
                    continue
                metadata = json.load(
                    open(os.path.join(subtopic_dir, qid, "metadata.json"))
                )
                title = metadata["title"].replace(" ", "_")
                title = title.replace("/", "_")
                html_file = os.path.join(subtopic_dir, qid, f"{title}.html")
                if not os.path.exists(html_file):
                    print(f"{html_file} does not exist")
                    continue
                with open(html_file) as f:
                    html = f.read()
                    tables = extract_html_tables_from_html(html)
                with open(output_filename, "w") as f:
                    for t in tables:
                        json.dump(t, f)
                        f.write("\n")
                print(f"Extracted {len(tables)} tables from {qid}")
