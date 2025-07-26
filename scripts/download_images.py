# Download images from the list of urls
import os
import json
from urllib.parse import urlparse, unquote
from pyWikiCommons import pyWikiCommons

if __name__ == "__main__":
    filename = "data/images_3-10.json"
    image_dir = "data/images"
    output_dir = "data/images_3-10"

    urls = {
        f.replace("File:", ""): 1
        for img_dir in [image_dir, output_dir]
        for f in os.listdir(img_dir)
        if os.path.isfile(os.path.join(img_dir, f))
    }
    print(f"Found {len(urls)} images")

    with open(filename, "r") as f:
        for line in f:
            data = json.loads(line)
            images = data["images"]
            if len(images) > 0:
                # download the list of images
                for image_url in images:
                    a = urlparse(image_url)
                    filename = os.path.basename(a.path)
                    if unquote(filename) in urls:
                        continue
                    # download the image from the url
                    try:
                        pyWikiCommons.download_commons_image(
                            f"File:{filename}", output_dir
                        )
                        urls[unquote(filename)] = 1
                    except Exception as e:
                        print(e)
                        print("Not found:", unquote(filename))
