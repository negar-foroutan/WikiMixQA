import os
import json

image_path = "/mnt/tamedia/wikitables"
image_folders = ["images_10+", "images_3-10"]


images = {}
for folder in image_folders:
    image_fullpath = os.path.join(image_path, folder)
    for image in os.listdir(image_fullpath):
        image_filename = os.path.join(image_fullpath, image)
        images[image] = image_filename
        print(image)

with open("data/path_to_images.json", "w") as fp:
    json.dump(images, fp)
