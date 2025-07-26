import pandas as pd
import json
import os
from tqdm import tqdm
from argparse import ArgumentParser
from typing import NamedTuple, Optional
from pydantic import BaseModel
from enum import Enum

from PIL import Image
from transformers import AutoTokenizer, AutoProcessor

import torch
from sklearn.metrics import classification_report
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from qwen_vl_utils import process_vision_info

from utils import define_prompt, load_html_tables, load_doc_images


class AnswerChoice(str, Enum):
    a = "A"
    b = "B"
    c = "C"
    d = "D"
    unable = "Unable to determine"


class Answer(BaseModel):
    explanation: str
    choice: AnswerChoice


guided_decoding_params = GuidedDecodingParams(
    choice=["A", "B", "C", "D", "Unable to determine"]
)

MAX_IMAGE_PROMPT = 2


class ModelRequestData(NamedTuple):
    llm: LLM
    tokenizer: AutoTokenizer
    table_image_prompt: str
    image_image_prompt: str
    table_table_prompt: str
    blind_prompt: str
    wikidoc_prompt: str


def load_internvl(
    model_name: str = "OpenGVLab/InternVL2-2B", num_gpus: int = 8
) -> ModelRequestData:
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        gpu_memory_utilization=0.95,
        enforce_eager=True,
        limit_mm_per_prompt={"image": MAX_IMAGE_PROMPT},
        tensor_parallel_size=num_gpus,
    )

    table_image_prompt = """
    Image:
    <image>

    HTML table:
    {html_table}

    """
    image_image_prompt = """
    Image1:
    <image>

    Image2:
    <image>

    """
    table_table_prompt = """
    HTML table1:
    {html_table1}


    HTML table2:
    {html_table2}

    """
    blind_prompt, wikidoc_prompt = "", ""
    # Add the question and answers to the prompt
    for p in [
        table_image_prompt,
        image_image_prompt,
        table_table_prompt,
        blind_prompt,
        wikidoc_prompt,
    ]:
        p += """
        Question: {generated_question}
        A: {answerA}
        B: {answerB}
        C: {answerC}
        D: {answerD}
        
        """
    table_image_prompt += define_prompt("oracle", n_charts=1, n_tables=1)
    image_image_prompt += define_prompt("oracle", n_charts=2, n_tables=0)
    table_table_prompt += define_prompt("oracle", n_charts=0, n_tables=2)

    blind_prompt += define_prompt("blind")
    wikidoc_prompt += define_prompt("wikidoc")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    return ModelRequestData(
        llm=llm,
        tokenizer=tokenizer,
        table_image_prompt=table_image_prompt,
        image_image_prompt=image_image_prompt,
        table_table_prompt=table_table_prompt,
        blind_prompt=blind_prompt,
        wikidoc_prompt=wikidoc_prompt,
    )


def load_qwen2_vl(
    model_name: str = "Qwen/Qwen2-VL-7B-Instruct", num_gpus: int = 8
) -> ModelRequestData:
    llm = LLM(
        model=model_name,
        dtype=torch.bfloat16,
        gpu_memory_utilization=0.80,
        max_model_len=32768,
        max_num_seqs=1,
        limit_mm_per_prompt={"image": MAX_IMAGE_PROMPT},
        tensor_parallel_size=num_gpus,
    )
    table_image_prompt = """
    HTML table:
    {html_table}

    """
    image_image_prompt = ""
    table_table_prompt = """
    HTML table1:
    {html_table1}


    HTML table2:
    {html_table2}

    """
    blind_prompt, wikidoc_prompt = "", ""
    # Add the question and answers to the prompt
    for p in [
        table_image_prompt,
        image_image_prompt,
        table_table_prompt,
        blind_prompt,
        wikidoc_prompt,
    ]:
        p += """
        Question: {generated_question}
        A: {answerA}
        B: {answerB}
        C: {answerC}
        D: {answerD}
        
        """
    table_image_prompt += define_prompt("oracle", n_charts=1, n_tables=1)
    image_image_prompt += define_prompt("oracle", n_charts=2, n_tables=0)
    table_table_prompt += define_prompt("oracle", n_charts=0, n_tables=2)

    blind_prompt += define_prompt("blind")
    wikidoc_prompt += define_prompt("wikidoc")

    processor = AutoProcessor.from_pretrained(model_name)

    return ModelRequestData(
        llm=llm,
        tokenizer=processor,
        table_image_prompt=table_image_prompt,
        image_image_prompt=image_image_prompt,
        table_table_prompt=table_table_prompt,
        blind_prompt=blind_prompt,
        wikidoc_prompt=wikidoc_prompt,
    )


def load_mllama(
    model_name: str = "meta-llama/Llama-3.2-11B-Vision-Instruct", num_gpus: int = 8
) -> ModelRequestData:
    llm = LLM(
        model=model_name,
        dtype=torch.bfloat16,
        gpu_memory_utilization=0.95,
        max_model_len=16384,
        max_num_seqs=16,
        limit_mm_per_prompt={"image": MAX_IMAGE_PROMPT},
        tensor_parallel_size=num_gpus,
        enforce_eager=True,
    )
    table_image_prompt = """
    <|image|><|image|><|begin_of_text|>
    HTML table:
    {html_table}

    """
    image_image_prompt = "<|image|><|image|><|begin_of_text|>"
    table_table_prompt = """
    <|begin_of_text|>
    HTML table1:
    {html_table1}


    HTML table2:
    {html_table2}

    """
    blind_prompt, wikidoc_prompt = "", ""
    # Add the question and answers to the prompt
    for p in [
        table_image_prompt,
        image_image_prompt,
        table_table_prompt,
        blind_prompt,
        wikidoc_prompt,
    ]:
        p += """
        Question: {generated_question}
        A: {answerA}
        B: {answerB}
        C: {answerC}
        D: {answerD}
        
        """
    table_image_prompt += define_prompt("oracle", n_charts=1, n_tables=1)
    image_image_prompt += define_prompt("oracle", n_charts=2, n_tables=0)
    table_table_prompt += define_prompt("oracle", n_charts=0, n_tables=2)

    blind_prompt += define_prompt("blind")
    wikidoc_prompt += define_prompt("wikidoc")

    return ModelRequestData(
        llm=llm,
        tokenizer=None,
        table_image_prompt=table_image_prompt,
        image_image_prompt=image_image_prompt,
        table_table_prompt=table_table_prompt,
        blind_prompt=blind_prompt,
        wikidoc_prompt=wikidoc_prompt,
    )


def load_image(image_file):
    return Image.open(image_file).convert("RGB")


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


def load_html_tables(qid_folder, qid):
    html_tables = []
    with open(f"{qid_folder}/{qid}/html_tables.json") as f:
        for line in f:
            data = json.loads(line)
            html_tables.append(data["html"])
    return html_tables


def do_inference(
    model, qid_to_path, generated_question, qa_type, model_name, explanation: bool = False
):

    html_table_ids = generated_question["table_ids"]
    images = generated_question["charts"]
    qid = generated_question["qid"]
    qid_path = qid_to_path[qid]
    html_tables = load_html_tables(qid_path, qid)
    question = generated_question["Question"]
    answerA = generated_question["A"]
    answerB = generated_question["B"]
    answerC = generated_question["C"]
    answerD = generated_question["D"]
    correct_answer = generated_question["Answer"]

    image_data = []
    if qa_type == "oracle":
        if len(html_table_ids) == 2:  # table-table question
            question = model.table_table_prompt.format(
                html_table1=html_tables[html_table_ids[0]],
                html_table2=html_tables[html_table_ids[1]],
                generated_question=question,
                answerA=answerA,
                answerB=answerB,
                answerC=answerC,
                answerD=answerD,
            )
        elif len(images) == 2:  # image-image question
            image_path_1 = os.path.join(qid_path, qid, "images", images[0])
            image_path_2 = os.path.join(qid_path, qid, "images", images[1])
            if "Qwen" in model_name:
                image_data.append({"type": "image", "image": image_path_1})
                image_data.append({"type": "image", "image": image_path_2})
            else:
                image_data.append(load_image(image_path_1))
                image_data.append(load_image(image_path_2))

            question = model.image_image_prompt.format(
                generated_question=question,
                answerA=answerA,
                answerB=answerB,
                answerC=answerC,
                answerD=answerD,
            )
        elif len(html_table_ids) == 1 and len(images) == 1:  # table-image question
            image_path = os.path.join(qid_path, qid, "images", images[0])
            if "Qwen" in model_name:
                image_data.append({"type": "image", "image": image_path})
            else:
                image_data.append(load_image(image_path))

            question = model.table_image_prompt.format(
                html_table=html_tables[html_table_ids[0]],
                generated_question=question,
                answerA=answerA,
                answerB=answerB,
                answerC=answerC,
                answerD=answerD,
            )
    elif qa_type == "blind":
        question = model.blind_prompt.format(
            generated_question=question,
            answerA=answerA,
            answerB=answerB,
            answerC=answerC,
            answerD=answerD,
        )
    elif qa_type == "wikidoc":
        image_files = load_doc_images(qid_path, qid)
        # truncate the images if they exceed the maximum limit
        if len(image_files) > MAX_IMAGE_PROMPT:
            image_files = image_files[:MAX_IMAGE_PROMPT]
        if "InternVL" in model_name:
            image_data = [load_image(image_file) for image_file in image_files]
            placeholders = "\n".join(
                f"Image-{i}: <image>\n" for i, _ in enumerate(image_files, start=1)
            )
            question = placeholders + model.wikidoc_prompt.format(
                generated_question=question,
                answerA=answerA,
                answerB=answerB,
                answerC=answerC,
                answerD=answerD,
            )
        elif "Qwen" in model_name:
            image_data = [
                {"type": "image", "image": image_file} for image_file in image_files
            ]
            question = model.wikidoc_prompt.format(
                generated_question=question,
                answerA=answerA,
                answerB=answerB,
                answerC=answerC,
                answerD=answerD,
            )
        elif "Llama" in model_name:
            question = model.wikidoc_prompt.format(
                generated_question=question,
                answerA=answerA,
                answerB=answerB,
                answerC=answerC,
                answerD=answerD,
            )
    if explanation:
        question += " Please provide a short explanation for the answer."
        sampling_params = SamplingParams(
            guided_decoding=GuidedDecodingParams(json=Answer.model_json_schema()),
            max_tokens=256,
        )
    else:
        sampling_params = SamplingParams(guided_decoding=guided_decoding_params)

    if "InternVL" in model_name:
        messages = [{"role": "user", "content": question}]
    elif "Qwen" in model_name:
        if len(image_data):
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        *image_data,
                        {"type": "text", "text": question},
                    ],
                },
            ]
            image_data, _ = process_vision_info(messages)
        else:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                    ],
                },
            ]
    if "Llama" in model_name:
        prompt = question
    else:
        prompt = model.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    if len(image_data) == 0:
        full_prompt = {
            "prompt": prompt,
        }
    else:
        full_prompt = {"prompt": prompt, "multi_modal_data": {"image": image_data}}
    
    # retry if JsonDecodeError occurs
    for i in range(5):
        try:
            outputs = model.llm.generate(full_prompt, sampling_params=sampling_params)
            pred = outputs[0].outputs[0].text
            if explanation:
                response = Answer.model_validate_json(pred)
                return response, correct_answer
            return pred, correct_answer
        except Exception as e:
            print(f"Error: {e}, retrying {i+1}")
            continue
    return None, correct_answer

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--input-file", type=str, required=True)
    argparser.add_argument("--output-dir", type=str, required=True)
    argparser.add_argument("--qid-to-path", type=str, required=True)
    argparser.add_argument(
        "--qa-type", type=str, required=True, choices=["wikidoc", "oracle", "blind"]
    )
    argparser.add_argument(
        "--model-name",
        type=str,
        required=True,
        default="OpenGVLab/InternVL2-2B",
        choices=[
            "OpenGVLab/InternVL2_5-1B-MPO",
            "OpenGVLab/InternVL2_5-8B-MPO",
            "OpenGVLab/InternVL2_5-8B",
            "OpenGVLab/InternVL2_5-26B-MPO",
            "OpenGVLab/InternVL2_5-38B-MPO",
            "OpenGVLab/InternVL2_5-78B",
            "OpenGVLab/InternVL2_5-78B-MPO",
            "meta-llama/Llama-3.2-11B-Vision-Instruct",
            "meta-llama/Llama-3.2-90B-Vision-Instruct",
            "OpenGVLab/InternVL2-2B",
            "OpenGVLab/InternVL2-Llama3-76B",
            "Qwen/Qwen2-VL-72B-Instruct",
            "Qwen/Qwen2-VL-7B-Instruct",
            "Qwen/Qwen2.5-VL-7B-Instruct",
            "Qwen/Qwen2.5-VL-72B-Instruct",
        ],
    )
    argparser.add_argument("--num-gpus", type=int, default=8)
    argparser.add_argument("--explanation", action="store_true")

    args = argparser.parse_args()
    # load the data
    df = pd.read_json(args.input_file)

    if "InternVL" in args.model_name:
        model = load_internvl(args.model_name, args.num_gpus)
    elif "Qwen" in args.model_name:
        model = load_qwen2_vl(args.model_name, args.num_gpus)
    elif "Llama" in args.model_name:
        model = load_mllama(args.model_name, args.num_gpus)
    else:
        raise ValueError("Invalid model name")

    qid_to_path = {}
    with open(args.qid_to_path) as f:
        for line in f:
            qid, path = line.split("\t")
            qid_to_path[qid] = path.strip()

    predictions = []
    ground_truths = []
    explanations = []
    for i, item in tqdm(df.iterrows(), total=len(df)):
        y_pred, y = do_inference(
            model, qid_to_path, item, args.qa_type, args.model_name, args.explanation
        )
        if args.explanation:
            explanations.append(y_pred.explanation)
            y_pred = y_pred.choice
        predictions.append(y_pred)
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
    if args.explanation:
        df["prediction_explanation"] = explanations
    # save the merged data in the output directory
    df.to_json(f"{output_dir}/merged_data.json", orient="records", lines=True)
