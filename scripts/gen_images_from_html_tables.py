import argparse
import os
import sys
import json
import pandas as pd
from selenium import webdriver
from selenium.webdriver.firefox.options import Options

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from core.wikitable_to_image import transform_html_id_text, html_to_img

opts = Options()
opts.add_argument("--headless")
driver = webdriver.Firefox(options=opts)


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
                html_filename = os.path.join(subtopic_dir, qid, "html_tables.json")
                if not os.path.exists(html_filename):
                    print(f"Skipping {qid}")
                    continue

                # read HTML tables
                with open(html_filename, "r") as f:
                    tables = pd.read_json(f.read(), lines=True)

                # create output folder
                output_folder = os.path.join(subtopic_dir, qid, "html_table_images")
                os.makedirs(output_folder, exist_ok=True)

                n_images = 0
                for idx in range(tables.shape[0]):
                    if tables.iloc[idx]["html"] is not None:
                        (
                            struc_tokens,
                            html_with_id,
                            list_cell_contents,
                            idx_count,
                        ) = transform_html_id_text(tables.iloc[idx]["html"])
                    if struc_tokens is None:
                        continue
                    im, bboxes = html_to_img(driver, html_with_id, idx_count)
                    if bboxes is None:
                        continue
                    im.save(os.path.join(output_folder, f"table_{idx}.png"))
                    n_images += 1
                print(f"{n_images} images generated from {len(tables)} tables of {qid}")
