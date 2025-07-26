import os
import shutil
import json
import pandas as pd

parent_folder = "images"
categories = json.load(open("data/category_taxonomy.json"))
# categories = {"Wikimedia": categories["Wikimedia"]}
qids = []
for topic, v in categories.items():
    for subtopic in v.keys():
        df = pd.read_json(f"data/{topic}_{subtopic}.json", lines=True)
        qids.extend(df["qid"])
d_qids = {i for i in qids}
print(f"Total qids: {len(d_qids)}")


def qid_to_image_paths(d_qids, filename, img_folder):
    with open(filename) as f:
        for line in f:
            data = json.loads(line)
            if data["qid"] not in d_qids:
                continue
            output_folder = os.path.join(parent_folder, data["qid"])
            os.makedirs(output_folder, exist_ok=True)
            for img in data["images"]:
                img_filename = os.path.basename(img).replace(".svg", ".png")
                img_path = os.path.join(img_folder, "File:" + img_filename)
                if os.path.exists(img_path):
                    output_img_path = os.path.join(output_folder, img_filename)
                    shutil.copy(img_path, output_img_path)


qid_to_image_paths(
    d_qids, "data/images_3-10.json", "/mnt/tamedia/wikitables/images_3-10/"
)
qid_to_image_paths(
    d_qids, "data/images_10+.json", "/mnt/tamedia/wikitables/images_10+/"
)
