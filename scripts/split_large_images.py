from PIL import Image
import argparse
import os
import json


stride = 768
overlap = 32

categories = json.load(open("data/category_taxonomy.json"))
categories = {"Wikimedia": categories["Wikimedia"]}

if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--input-dir", type=str, required=True)
    args = argparser.parse_args()

    for topic, v in categories.items():
        topic_dir = os.path.join(args.input_dir, topic)
        for subtopic in v.keys():
            subtopic_dir = os.path.join(topic_dir, subtopic)
            for qid in os.listdir(subtopic_dir):
                metadata = json.load(
                    open(os.path.join(subtopic_dir, qid, "metadata.json"))
                )
                title = metadata["title"].replace(" ", "_")
                qid_dir = os.path.join(subtopic_dir, qid)
                output_dir = os.path.join(qid_dir, "html_images")
                # if the html_images folder already exists, skip
                if os.path.exists(output_dir):
                    print(f"Skipping {qid_dir}")
                    continue
                html_image_file = os.path.join(qid_dir, f"{title}.jpg")
                if not os.path.exists(html_image_file):
                    print(f"{html_image_file} does not exist")
                    continue
                try:
                    img = Image.open(html_image_file)
                except Exception as e:
                    print(f"Could not open {html_image_file}")
                    print(e)
                    # remove the file
                    os.remove(html_image_file)
                    continue
                imgwidth, imgheight = img.size
                chunks = [
                    (i, i + stride) for i in range(0, imgheight, stride - overlap)
                ]
                output_dir = os.path.join(qid_dir, "html_images")
                os.makedirs(output_dir, exist_ok=True)
                print(f"Splitting {html_image_file} into {len(chunks)} chunks")
                for i, (start, end) in enumerate(chunks):
                    img_chunk = img.crop((0, start, imgwidth, end))
                    img_chunk.save(os.path.join(output_dir, f"{title}_{i}.jpg"))
