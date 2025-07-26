from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from transformers import AutoModel, AutoTokenizer

import torch
import argparse
import json
import os
import pandas as pd
from PIL import Image
import numpy as np

categories = json.load(open("data/category_taxonomy.json"))
model_name = "gpt-4-turbo"


def list_map_char(dataset_dir: str):
    map_charts = []
    for topic, v in categories.items():
        for subtopic in v.keys():
            if not os.path.isdir(f"{dataset_dir}/{topic}/{subtopic}"):
                continue
            for qid in os.listdir(f"{dataset_dir}/{topic}/{subtopic}"):
                if not os.path.exists(
                    f"{dataset_dir}/{topic}/{subtopic}/{qid}/{model_name}.json"
                ):
                    continue
                df = pd.read_json(
                    f"{dataset_dir}/{topic}/{subtopic}/{qid}/{model_name}.json",
                    lines=True,
                )
                for _, data in df.iterrows():
                    if "error" in data:
                        continue
                    try:
                        output = json.loads(
                            data["choices"][0]["message"]["content"], strict=False
                        )
                    except json.decoder.JSONDecodeError:
                        print(data["choices"][0]["message"]["content"])
                    except KeyError:
                        print(data)
                    if "type" in output:
                        if output["type"] == "map":
                            filename = data["filename"]
                            path_to_image = f"{dataset_dir}/{topic}/{subtopic}/{qid}/images/{filename}"
                            if os.path.exists(path_to_image):
                                map_charts.append(
                                    {
                                        "filename": path_to_image,
                                        "data": output["data"],
                                    }
                                )
    return map_charts


def detect_map_with_text(
    model_name: str, device: str, map_charts: list, max_new_tokens: int
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    if "llava" in model_name:
        processor = LlavaNextProcessor.from_pretrained(model_name)
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        prompt = "[INST] <image>\nPlease extract all the text from the image[/INST]"
    elif "MiniCPM" in model_name:
        processor = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        prompt = "Please extract all the text from the image."
    else:
        raise ValueError("Model not supported")

    model.to(device)
    model.eval()

    for chart in map_charts:
        image_filename = chart["filename"]
        image = Image.open(image_filename)
        image = image.convert("RGB")
        if "llava" in model_name:
            inputs = processor(prompt, image, return_tensors="pt").to(device)
            output = model.generate(**inputs, max_new_tokens=max_new_tokens)
            ouput_text = processor.decode(output[0], skip_special_tokens=True)
        elif "MiniCPM" in model_name:
            msgs = [{"role": "user", "content": prompt}]
            res = model.chat(
                image=image,
                msgs=msgs,
                tokenizer=processor,
                sampling=True,  # if sampling=False, beam_search will be used by default
                temperature=0.7,
                # system_prompt='' # pass system_prompt if needed
            )
            ouput_text = ""
            for new_text in res:
                ouput_text += new_text

        # save in file
        qid_folder = os.path.dirname(os.path.dirname(image_filename))
        print(f"Saving metadata for {qid_folder}")
        with open(
            f"{qid_folder}/map_chart_metadata-{os.path.basename(model_name)}.json", "a"
        ) as f:
            json.dump(
                {
                    "filename": os.path.basename(image_filename),
                    "description": chart["data"],
                    "extracted_text": ouput_text,
                },
                f,
            )
            f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir", type=str, default="/mnt/tamedia/wikitables/final_new_2"
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--model_name", type=str, default="llava-hf/llava-v1.6-mistral-7b-hf"
    )
    parser.add_argument("--max_new_tokens", type=int, default=300)
    args = parser.parse_args()

    map_charts = list_map_char(args.dataset_dir)
    detect_map_with_text(args.model_name, args.device, map_charts, args.max_new_tokens)
