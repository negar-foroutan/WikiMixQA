import openai
from argparse import ArgumentParser
import pandas as pd
import json
import base64
import requests
import os

client = openai.OpenAI()
openai.api_key = os.environ["OPENAI_API_KEY"]
openai_model = "gpt-4-turbo"  
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {openai.api_key}",
}

####
# Prompt used with gpt-4-vision-preview
####
# prompt = """
# Is the image a chart?
# A chart (sometimes known as a graph) is a graphical representation for data visualization, in which "the data is represented by symbols, such as bars in a bar chart, lines in a line chart, or slices in a pie chart".[1] A chart can represent tabular numeric data, functions or some kinds of quality structure and provides different info.
# If yes, please specify the type of chart between data chart, maps or other.
# A data chart is a type of diagram or graph, that organizes and represents a set of numerical or qualitative data.
# Maps that are adorned with extra information (map surround) for a specific purpose are often known as charts, such as a nautical chart or aeronautical chart, typically spread over several map sheets.
# Then extract all information from the chart.
# You are communicating with an API, not a user. Begin all AI responses with the character ‘{’ to produce valid JSON. Here is an example:
# {
#     "chart": "yes",
#     "type": "data chart",
#     "data": "information extracted from the image"
# }
# """

prompt = """
Is the image a chart? 
A chart (sometimes known as a graph) is a graphical representation for data visualization, in which "the data is represented by symbols, such as bars in a bar chart, lines in a line chart, or slices in a pie chart". A chart can represent tabular numeric data, functions or some kinds of quality structure and provides different info.
If yes, please specify the type of chart between data chart, maps or other. 
A data chart is a type of diagram or graph, that organizes and represents a set of numerical or qualitative data.
Maps that are adorned with extra information (map surround) for a specific purpose are often known as charts, such as a nautical chart or aeronautical chart, typically spread over several map sheets.
Then extract the most relevant information from the chart in less than 200 words.
You are communicating with an API, not a user. Begin all AI responses with the character ‘{’ to produce valid JSON. Here is an example:
{  
    "chart": "yes",
    "type": "data chart",
    "data": "information extracted from the image"
}
"""


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_chart_information(base64_image: str) -> str:
    payload = {
        "model": openai_model,
        "response_format": {
            "type": "json_object"
        },  # not used with gpt-4-vision-preview
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ],
        "max_tokens": 300,
    }

    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
    )
    return response.json()


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--data-dir", type=str, required=True, default="data")
    argparser.add_argument("--topic", type=str, required=True)
    argparser.add_argument("--subtopic", type=str, required=True)
    argparser.add_argument("--output-dir", type=str, required=True, default="final")
    argparser.add_argument("--chart", action="store_true")
    args = argparser.parse_args()

    input_filepath = os.path.join(
        args.data_dir, f"{args.topic}_{args.subtopic}_chart_stats.json"
    )
    df = pd.read_json(input_filepath, lines=True)
    print(df.head())
    print(df.info())

    with open("data/path_to_images.json", "r") as f:
        path_to_images = json.load(f)

    answers = []
    for i, row in df.iterrows():
        images = row["images"]
        qid = row["qid"]
        n_charts = row["n_charts"]
        if args.chart and n_charts == 0:
            print(f"Skipping {qid} as it does not have any charts")
            continue

        output_filename = os.path.join(
            args.output_dir, f"{args.topic}/{args.subtopic}/{qid}/{openai_model}.json"
        )
        if os.path.exists(output_filename):
            with open(output_filename, "r") as f:
                lines = f.readlines()
                errors = [True for line in lines if "error" in line]
            if len(errors) < len(lines):
                print(f"Skipping {qid}")
                continue

        charts = []
        for image in images:
            filename = image["filename"]
            try:
                image_path = path_to_images[f"File:{filename}"]
                if not os.path.exists(image_path):
                    print(f"Could not find image for {filename}")
                    continue
                base64_image = encode_image(image_path)
                prediction = get_chart_information(base64_image)
                charts.append(
                    {
                        "filename": filename,
                        **prediction,
                    }
                )
            except Exception as e:
                print(f"Could not get prediction for {filename}: {e}")

        with open(output_filename, "w") as f:
            for chart in charts:
                json.dump(chart, f)
                f.write("\n")
        print(f"Saved {qid}")
    print("Done")
