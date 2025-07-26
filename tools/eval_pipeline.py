import json
import os

import requests

from config.config import DIR_WIKI_TABLES
from generation.question_generation import get_modality_ranking
from generation.utils import get_table_content, encode_image
from tools.utils import get_id_to_dir, CustomError

OPENAI_MODELS = ['gpt-4-turbo', "gpt-4o"]
openai_api_key = ""
openai_headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {openai_api_key}"
    }

def get_instruction(with_mod=False, with_page=False):
    instruction = ""
    if not with_mod and not with_page:
        instruction += "Answer the following multiple-choice question. Please explain how to reach the correct answer step-by-step.\n"
    elif with_page and with_mod:
        instruction += "You are provided with some document pages, some modalities from these pages e.g. charts and tables, and a multiple-choice question based on these modalities. Your task is to answer the multiple-choice question.\n"
        instruction += "First identify and combine the common information between the given modalities with the help of document pages. Then summarize this common information clearly. Highlight any trends, patterns, or key points that are present in the modalities.\n"
        instruction += "As your response, answer the multiple-choice question. Please explain how to reach the correct answer from the provided information and modalities step-by-step.\n"
    elif with_page:
        instruction += "You are provided with some document pages and a multiple-choice question based on the modalities of these pages e.g. charts and tables. Your task is to answer the multiple-choice question.\n"
        instruction += "First identify and combine the common information between the given pages. Then summarize this common information clearly. Highlight any trends, patterns, or key points that are present in the modalities.\n"
        instruction += "As your response, answer the multiple-choice question. Please explain how to reach the correct answer from the provided information and modalities step-by-step.\n"
    elif with_mod:
        instruction += "You are provided with some modalities e.g. charts and tables and a multiple-choice question based on these modalities. Your task is to answer the multiple-choice question.\n"
        instruction += "First identify and combine the common information between the given modalities. Then summarize this common information clearly. Highlight any trends, patterns, or key points that are present in the modalities.\n"
        instruction += "As your response, answer the multiple-choice question. Please explain how to reach the correct answer from the provided information and modalities step-by-step.\n"
    else:
        raise CustomError("Invalid arguments")

    instruction += """
    You are communicating with an API, not a user. The question will be given you in the following json format:
    {
        "Question": "<question>",
         "A": "<option1> ",
         "B": "<option2>",
         "C": "<option3>",
         "D": "<option4>"
    }
     
    Begin all AI responses with the character '{' to produce valid JSON. Here is an example:
    {
         "Answer": "<correct_option>",
         "Explanation": "<explanation>"
    }   
    
    If generating such a question is not possible, say it's not possible in the following format:
    {   
     "Answer": "not possible"
    }
    """
    return instruction


def prepare_mcq_generation_data(question,
                                page_data,
                                instruction_prompt,
                                with_mod=False,
                                with_page=False,
                                with_desc=False,
                                page_text=False):
    data = {}
    if with_mod:
        tables_records = question["table_ids"]
        tables_content = [get_table_content(page_data["tables"][t], with_desc=with_desc) for t in tables_records]
        chart_paths = question["charts"]
        table_ids = question["table_ids"]
        charts = []
        for path in chart_paths:
            path = os.path.join(page_data["page_folder"], "images", path)
            base64_image = encode_image(path)
            charts.append(base64_image)
        data = {
            "charts": charts,
            "chart_path": chart_paths,
            "tables": tables_content,
            "table_ids": table_ids,
        }
    if with_page:
        pages = []
        for path in page_data["pages"]:
            base64_image = encode_image(path)
            pages.append(base64_image)
        data["pages"] = pages

    if with_desc:
        data['chart_explanations'] = page_data['chart_explanations']
    data["prompt"] = instruction_prompt
    if "Q3" in question["generated_question"]:
        question_data = question["generated_question"]["Q3"]
    else:
        question_data = question["generated_question"]
    try:
        data["question"] = {
            "Question": question_data["Question"],
            "A": question_data["A"],
            "B": question_data["B"],
            "C": question_data["C"],
            "D": question_data["D"]
        }
    except Exception as e:
        raise CustomError(f"Question key error: {e}")
    data["answer"] = question_data["Answer"]
    data["answer_explanation"] = question_data["Explanation"]
    data["wikidata_id"] = question["wikidata_id"]
    if page_text:
        data["page_text"] = page_data["page_text"]
    return data

def get_page_data(question, wiki_page_stats_map, with_desc=False, index=None,
                  with_page=False, single_page=False, page_text=False):
    wikidata_id = question['wikidata_id']
    page_folder = wiki_page_stats_map[wikidata_id]

    chart_names = question['charts']
    if with_desc:
        table_file = os.path.join(page_folder, "html_tables_with_desc.json")
    else:
        table_file = os.path.join(page_folder, "html_tables.json")
    charts = []
    if len(chart_names) != 0:
        charts = [os.path.join(page_folder, "images", chart_name) for chart_name in chart_names]
    data = {"index": index, "charts": charts, "tables": {}, "wikidata_id": wikidata_id}
    chart_explanations = []
    if len(charts) > 0:
        chart_explanation_path = os.path.join(page_folder, "gpt-4-turbo.json")
        with open(chart_explanation_path, "r") as f:
            all_exp = [json.loads(row) for row in f.readlines()]
        chart_explanations_dict = {}
        for j in all_exp:
            if 'choices' in j:
                descs = [
                    json.loads(ch['message']['content'])['data'] if isinstance(ch['message']['content'], str) else
                    ch['message']['content']['data'] for ch in j['choices']]
                filtered_decs = []
                for d in descs:
                    if d is not None and d != "":
                        filtered_decs.append(d)
                if len(filtered_decs) > 0:
                    chart_explanations_dict[j['filename']] = "\n".join(filtered_decs)
            else:
                print("faced error in finding chart description")
        for chart in chart_names:
            if chart in chart_explanations_dict.keys():
                chart_explanations.append(chart + " chart's description: " + chart_explanations_dict[chart])
            else:
                chart_explanations.append("")

    data['chart_explanations'] = chart_explanations
    if not os.path.exists(table_file):
        raise CustomError(f"File not found: {table_file}")
    with open(table_file, "r") as f:
        data["tables"] = [json.loads(row) for row in f.readlines()]

    table_ranking_file = os.path.join(page_folder, "table_to_table_bge-reranker-v2-m3.json")
    if os.path.exists(table_ranking_file):
        with open(table_ranking_file, "r") as f:
            data["table_ranking"] = get_modality_ranking([json.loads(row) for row in f.readlines()])
    image_ranking_file = os.path.join(page_folder, "image_to_image_bge-reranker-v2-m3.json")

    if os.path.exists(image_ranking_file):
        with open(image_ranking_file, "r") as f:
            data["image_ranking"] = get_modality_ranking([json.loads(row) for row in f.readlines()])

    image_table_ranking_file = os.path.join(page_folder, "table_to_image_bge-reranker-v2-m3.json")
    if os.path.exists(image_table_ranking_file):
        with open(image_table_ranking_file, "r") as f:
            data["image_table_ranking"] = get_modality_ranking([json.loads(row) for row in f.readlines()])
    if with_page:
        if single_page:
            files = os.listdir(page_folder)
            for file in files:
                if file.endswith('.jpg'):
                    data['pages'] = [os.path.join(page_folder, file)]
                    break
            if 'pages' not in data:
                raise CustomError("No page has been found")
        else:
            try:
                page_image_path = os.path.join(page_folder, 'html_images')
                files = os.listdir(page_image_path)
            except Exception as e:
                raise CustomError(f"No page has been found: {e}")

            data['pages'] = [os.path.join(page_image_path, file) for file in files]
    data["page_folder"] = page_folder
    if page_text:
        with open(os.path.join(page_folder, "wiki.txt"), 'r') as f:
            data['page_text'] = f.read()

    return data

def get_data(questions, model_name, wiki_page_stats_map, instruction_prompt,
             with_mod=False, with_page=False, with_desc=False, single_page=False,
             page_text=False):
    data_qs = []
    if model_name in OPENAI_MODELS:
        for i, question in enumerate(questions):
            try:
                page_data = get_page_data(question, wiki_page_stats_map, with_desc=with_desc, index=i,
                                          with_page=with_page, single_page=single_page, page_text=page_text)
                data = prepare_mcq_generation_data(question, page_data, instruction_prompt, with_mod=with_mod,
                                                   with_page=with_page, with_desc=with_desc, page_text=page_text)
            except CustomError as e:
                print(f"Error at {question['wikidata_id']}: ", e)
                continue

            data_qs.append(data)
    else:
        raise CustomError("Model not supported!")

    return data_qs


def get_response(data, model_name):
    if model_name in OPENAI_MODELS:
        message_text = data["prompt"] + "\n"
        message_text += "Question:\n"
        message_text += json.dumps(data["question"]) + "\n"
        if "tables" in data and len(data["tables"]) > 0:
            message_text += "Tables:\n"
            for table_content in data["tables"]:
                message_text += table_content + "\n"

        if "chart_explanations" in data and len(data["chart_explanations"]) > 0:
            for chart_explanation in data["chart_explanations"]:
                message_text += chart_explanation + "\n"

        if "page_text" in data:
            message_text += "Pages' text:\n" + data["page_text"] + "\n"
        content = []
        content.append({
            "type": "text",
            "text": message_text
        })
        if "pages" in data:
            for page in data["pages"]:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{page}"
                    }
                })
        if "charts" in data:
            for chart in data["charts"]:
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

        req = {
            "model": model_name,
            "messages": messages,
        }
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=openai_headers, json=req)

        try:
            response = response.json()
        except Exception as e:
            raise CustomError(f"Invalid response: {e}")
        if "error" in response:
            print("Error: ", response["error"])
            raise CustomError("Error")

        response_text = response["choices"][0]["message"]["content"]

        try:
            json_response = json.loads(response_text)
        except Exception as e:
            raise CustomError(f"Non-json response: {e}")
        return json_response
    else:
        raise CustomError("Invalid model")

def get_accuracy(generations):
    false_generations = []
    for generation in generations:
        if generation["answer"] != generation["generated_answer"]["Answer"]:
            false_generations.append(generation["wikidata_id"])
    return 1 - len(false_generations) / len(generations), false_generations

def eval_mcq(questions, model_name, with_mod=False, with_page=False, with_desc=False, output_file=None, single_page=False, page_text=False):
    wiki_page_stats_map = get_id_to_dir(DIR_WIKI_TABLES)

    instruction_prompt = get_instruction(with_mod=with_mod, with_page=with_page)
    output = []
    data_qs = get_data(questions, model_name, wiki_page_stats_map, instruction_prompt, with_mod=with_mod,
                       with_page=with_page, with_desc=with_desc, single_page=single_page, page_text=page_text)
    for i, (question, data) in enumerate(zip(questions, data_qs)):
        print('evaluating ', question['wikidata_id'])
        try:
            response = get_response(data, model_name)
        except CustomError as e:
            print(f"Error: {e}")
            continue

        if "Answer" not in response:
            print(f"Invalid response: {response}")
            continue
        if with_mod:
            chart_paths = [chart_path.split("/")[-1] if chart_path != "" else "" for chart_path in data["chart_path"]]
            output_record = {"wikidata_id": data["wikidata_id"], "table_ids": data["table_ids"],
                             "charts": chart_paths, "question": data["question"], "answer": data["answer"],
                             "answer_explanation": data["answer_explanation"], "generated_answer": response}
        else:
            output_record = {"wikidata_id": question["wikidata_id"], "table_ids": question["table_ids"],
                             "charts": question["charts"], "question": data["question"], "answer": data["answer"],
                             "answer_explanation": data["answer_explanation"], "generated_answer": response}
        output.append(output_record)

        if i % 10 == 0:
            print("i: ", i)
            with open(output_file, "a+") as f:
                json.dump(output, f, indent=4)

    with open(output_file, "w") as f:
        json.dump(output, f, indent=4)

    accuracy, false_generations = get_accuracy(output)
    print("Generated Number: ", len(output))
    print("Accuracy: ", accuracy)
    print("False Generations: ", false_generations)

    return output, accuracy, false_generations

