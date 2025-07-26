from argparse import ArgumentParser
import pandas as pd
import json
from tqdm import tqdm
from sklearn.metrics import classification_report
import time
import os
from google.api_core.exceptions import ResourceExhausted
from utils import define_prompt, encode_image
from openai import OpenAI
from pydantic import BaseModel

api_key = ""
client = OpenAI(api_key=api_key)

MAX_IMAGE_PROMPT = 15

response_schema = {
    "type": "STRING",
    "enum": ["A", "B", "C", "D", "Unable to determine"],
}

question_body_format = """
        Question: {generated_question}
        A: {answerA}
        B: {answerB}
        C: {answerC}
        D: {answerD}
        
        """

table_table_prompt = """
    HTML table1:
    {html_table1}


    HTML table2:
    {html_table2}

    """

table_image_prompt = """
    HTML table:
    {html_table}

    """

class Answer(BaseModel):
    final_answer: str
    
def load_html_tables(qid_folder, qid):
    html_tables = []
    with open(f"{qid_folder}/{qid}/html_tables.json") as f:
        for line in f:
            data = json.loads(line)
            html_tables.append(data['html'])
    return html_tables

def load_doc_images(qid_folder, qid):
    # load all images from the html_images folder
    images = []
    input_folder = f"{qid_folder}/{qid}/html_images"
    if not os.path.exists(input_folder):
        print(f"Path does not exist: {input_folder}")
        return images
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg"):
            images.append(os.path.join(input_folder, filename))
    return images

def call_api_with_retries(api_call, max_retries=5):
    retries = 0
    while retries < max_retries:
        try:
            # Make the API call
            return api_call()
        except ResourceExhausted as e:
            retries += 1
            wait_time = 2 ** retries  # Exponential backoff
            print(f"ResourceExhausted error encountered. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)

    raise Exception("Max retries reached. ResourceExhausted error persists.")

def do_inference(qid_to_path, generated_question, qa_type):

    html_table_ids = generated_question['table_ids']
    images = generated_question['charts']
    qid = generated_question['qid']
    qid_path = qid_to_path[qid]
    html_tables = load_html_tables(qid_path, qid)
    question = generated_question['Question']
    answerA = generated_question['A']
    answerB = generated_question['B']
    answerC = generated_question['C']
    answerD = generated_question['D']
    correct_answer = generated_question['Answer']
    instruction = define_prompt(qa_type, n_charts=len(images), n_tables=len(html_table_ids))
    prompt = instruction + question_body_format.format(
        generated_question=question,
        answerA=answerA,
        answerB=answerB,
        answerC=answerC,
        answerD=answerD
    )
    image_data = []
    content = []
    if qa_type == "oracle":
        if len(html_table_ids) == 2: # table-table question
            question = prompt + table_table_prompt.format(html_table1=html_tables[html_table_ids[0]],
                                                html_table2=html_tables[html_table_ids[1]])
        elif len(images) == 2: # image-image question
            question = prompt
            image_path_1 = os.path.join(qid_path, qid, "images", images[0])
            image_data.append(encode_image(image_path_1))
            image_path_2 = os.path.join(qid_path, qid, "images", images[1])
            image_data.append(encode_image(image_path_2))
        
        elif len(html_table_ids) == 1 and len(images) == 1: # table-image question
            question = prompt + table_image_prompt.format(html_table=html_tables[html_table_ids[0]])
            image_path = os.path.join(qid_path, qid, "images", images[0])
            image_data.append(encode_image(image_path))

    elif qa_type == "blind":
        question = prompt
    elif qa_type == "wikidoc":
            image_files = load_doc_images(qid_path, qid)
            image_data = [encode_image(image_file) for image_file in image_files]
            question = prompt

    content.append({
                "type": "text",
                "text": question
            })
    for chart in image_data:
        content.append({
            "type": "image_url",
            "image_url": {
                 "url": f"data:image/jpeg;base64,{chart}"
            }
        })
            
    messages = [{
        "role": "user",
        "content": content
    }]
    try:
        completion = client.beta.chat.completions.parse(model=args.model_name, messages=messages, response_format=Answer)
        final_answer = completion.choices[0].message.parsed
        if final_answer.final_answer in ["A", "B", "C", "D", "Unable to determine"]:
            pred = final_answer.final_answer
        else:
            pred = "Error"
            print(f"Format Error in processing response for qid: {qid} ({final_answer.final_answer})")
    except Exception as e:
        pred = "Error"
        print(f"API Error in processing response for qid: {qid} ({e})")
    

    return pred, correct_answer

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--input-file", type=str, required=True)
    argparser.add_argument("--output-dir", type=str, required=True)
    argparser.add_argument("--qid-to-path", type=str, required=True)
    argparser.add_argument("--qa-type", type=str,  choices=["wikidoc", "oracle", "blind"],  required=True)
    argparser.add_argument("--model-name", type=str, choices=["gpt-4o"], required=True)
    args = argparser.parse_args()


    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}
    df = pd.read_json(args.input_file)

    # create directory if it does not exist
    output_dir = f"{args.output_dir}/{args.model_name}/{args.qa_type}"
    os.makedirs(output_dir, exist_ok=True)

    qid_to_path = {}
    with open(args.qid_to_path) as f:
        for line in f:
            qid, path = line.split('\t')
            qid_to_path[qid] = path.strip()
                       
    predictions = []
    ground_truths = []
    errors = []

    if os.path.exists(f"{output_dir}/predictions.json"):
        with open(f"{output_dir}/predictions.json") as f:
            predictions = json.load(f)
        with open(f"{output_dir}/ground_truths.json") as f:
            ground_truths = json.load(f)
        print(f"Loaded previous predictions: {len(predictions)}")

    for i, item in tqdm(df.iterrows(), total=len(df)):
        if i < len(predictions):
            continue
        y_pred, y = do_inference(qid_to_path, item, args.qa_type)
        if y_pred == "Error":
            errors.append(item['qid'])
        predictions.append(y_pred)
        ground_truths.append(y)
        if i % 20 == 0:
            with open(f"{output_dir}/predictions.json", "w") as f:
                f.write(json.dumps(predictions))
            with open(f"{output_dir}/ground_truths.json", "w") as f:
                f.write(json.dumps(ground_truths))

    print("****************************")
    print(f"Format Errors: {errors}\n")
    print("****************************")
    report = classification_report(ground_truths, predictions)
    print(report)

    

    # save the report in the output directory
    with open(f"{output_dir}/classification_report.txt", "w") as f:
        f.write(report)
    # save the predictions in the output directory
    with open(f"{output_dir}/predictions.json", "w") as f:
        json.dump(predictions, f)
    # save the ground truth in the output directory
    with open(f"{output_dir}/ground_truths.json", "w") as f:
        json.dump(ground_truths, f)

    # merge the predictions and ground truth with the input data
    df["predictions"] = predictions
    df["ground_truth"] = ground_truths
    # save the merged data in the output directory
    df.to_json(f"{output_dir}/merged_data.json", orient="records", lines=True)