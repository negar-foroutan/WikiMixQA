from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import os
import torch
from glob import glob
from tqdm import tqdm

device = "cuda:2"
batch_size = 64


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path, processor, extensions=["jpg", "png", "jpeg"]):
        self.processor = processor

        extensions = [ext.lower() for ext in extensions]
        extensions += [ext.upper() for ext in extensions]

        # Use glob to get a list of file paths that match the extensions
        self.image_paths = []
        for ext in extensions:
            pattern = os.path.join(folder_path, f"*.{ext}")
            self.image_paths.extend(glob(pattern, recursive=True))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx])
            inputs = self.processor(images=image, return_tensors="pt")
            return inputs, self.image_paths[idx]
        except Exception as e:
            print(e)
            print(f"Could not process {self.image_paths[idx]}!")
            return None


processor = AutoImageProcessor.from_pretrained(
    "facebook/dinov2-base-imagenet1k-1-layer"
)
model = AutoModelForImageClassification.from_pretrained(
    "facebook/dinov2-base-imagenet1k-1-layer"
)
model.to(device)

# Call the function to list images with the specified extensions in the folder
image_dst = ImageDataset("data/images_3-10", processor)
image_loader = torch.utils.data.DataLoader(
    image_dst,
    batch_size=batch_size,
    shuffle=False,
    num_workers=32,
    collate_fn=collate_fn,
)


with open("data/image_labels_3-10.tsv", "w") as f:
    with torch.no_grad():
        for batch in tqdm(image_loader):
            img_inputs, img_filenames = batch
            img_inputs = img_inputs.to(device)
            outputs = model(pixel_values=img_inputs["pixel_values"].squeeze(1))
            logits = outputs.logits
            predicted_class_idx = logits.argmax(-1)
            for k, img_filename in enumerate(img_filenames):
                label = model.config.id2label[predicted_class_idx[k].item()]
                f.write(f"{img_filename}\t{label}\n")
