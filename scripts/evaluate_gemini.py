from argparse import ArgumentParser
import pandas as pd
import json
from tqdm import tqdm
from sklearn.metrics import classification_report
import time
import os
from vertexai.generative_models import GenerativeModel, GenerationConfig, Part
from google.api_core.exceptions import ResourceExhausted
from utils import define_prompt, load_html_tables, load_doc_images


response_schema = {
    "type": "STRING",
    "enum": ["A", "B", "C", "D", "Unable to determine"],
}


def call_api_with_retries(api_call, max_retries=5):
    retries = 0
    while retries < max_retries:
        try:
            # Make the API call
            return api_call()
        except ResourceExhausted as e:
            retries += 1
            wait_time = 2**retries  # Exponential backoff
            print(
                f"ResourceExhausted error encountered. Retrying in {wait_time} seconds..."
            )
            time.sleep(wait_time)

    raise Exception("Max retries reached. ResourceExhausted error persists.")


def load_modalities(qa_type, topic, subtopic, qid, charts, table_ids):

    if qa_type == "wikidoc":
        return load_doc_images(topic, subtopic, qid)
    if qa_type == "oracle":
        modalities = []
        if len(table_ids) > 0:
            html_tables = load_html_tables(topic, subtopic, qid)
        if len(charts) == 2:
            for i in range(2):
                modalities.append(
                    Part.from_uri(
                        f"gs://wiki-chart-tables/{topic}/{subtopic}/{qid}/images/{charts[0]}",
                        mime_type="image/png",
                    )
                )
        elif len(table_ids) == 2:
            for i in range(2):
                modalities.append(Part.from_text(html_tables[table_ids[i]]))
        else:
            modalities.append(
                Part.from_uri(
                    f"gs://wiki-chart-tables/{topic}/{subtopic}/{qid}/images/{charts[0]}",
                    mime_type="image/png",
                )
            )
            modalities.append(Part.from_text(html_tables[table_ids[0]]))
    elif qa_type == "blind":
        modalities = []
    else:
        raise ValueError("Invalid QA type")

    return modalities


def do_inference(model, item, qa_type):
    charts = item["charts"]
    table_ids = item["table_ids"]
    topic = item["topic"]
    subtopic = item["subtopic"]
    qid = item["qid"]
    modalities = load_modalities(qa_type, topic, subtopic, qid, charts, table_ids)
    prompt = define_prompt(qa_type, n_charts=len(charts), n_tables=len(table_ids))

    modalities.append(Part.from_text(f"Question: {item['Question']}"))
    modalities.append(Part.from_text(f"A: {item['A']}"))
    modalities.append(Part.from_text(f"B: {item['B']}"))
    modalities.append(Part.from_text(f"C: {item['C']}"))
    modalities.append(Part.from_text(f"D: {item['D']}"))
    modalities.append(Part.from_text(prompt))
    correct_answer = item["Answer"]
    response = call_api_with_retries(
        lambda: model.generate_content(
            modalities,
            generation_config=GenerationConfig(
                response_mime_type="application/json", response_schema=response_schema
            ),
        ),
        max_retries=10,
    )
    return response.text, correct_answer


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--input-file", type=str, required=True)
    argparser.add_argument("--output-dir", type=str, required=True)
    argparser.add_argument(
        "--qa-type", type=str, required=True, choices=["wikidoc", "oracle", "blind"]
    )
    argparser.add_argument(
        "--model-name",
        type=str,
        required=True,
        choices=[
            "gemini-2.0-pro-exp-02-05",
            "gemini-2.0-flash-001",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-1.5-flash-8b",
        ],
    )
    args = argparser.parse_args()
    # load the data
    df = pd.read_json(args.input_file)

    # load the model
    model = GenerativeModel(args.model_name)

    predictions, ground_truths = [], []
    for i, item in tqdm(df.iterrows(), total=len(df)):
        y_pred, y = do_inference(model, item, args.qa_type)
        predictions.append(json.loads(y_pred))
        ground_truths.append(y)

    report = classification_report(ground_truths, predictions)
    print(report)

    # create directory if it does not exist
    output_dir = f"{args.output_dir}/{args.model_name}/{args.qa_type}"
    os.makedirs(output_dir, exist_ok=True)

    # save the report in the output directory
    with open(f"{output_dir}/classification_report.txt", "w") as f:
        f.write(report)
    # save the predictions in the output directory
    with open(f"{output_dir}/predictions.json", "w") as f:
        json.dump(predictions, f)
    # save the ground truth in the output directory
    with open(f"{output_dir}/ground_truth.json", "w") as f:
        json.dump(ground_truths, f)

    # merge the predictions and ground truth with the input data
    df["predictions"] = predictions
    df["ground_truth"] = ground_truths
    # save the merged data in the output directory
    df.to_json(f"{output_dir}/merged_data.json", orient="records", lines=True)
