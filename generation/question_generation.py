import os
import json
import random
import requests
import argparse

from config.config import DIR_WIKI_TABLES
from tools.interpretation import get_filtered_modalities, similarity_filter_group_by_topic
from tools.utils import get_id_to_dir
from generation.utils import encode_image, get_table_content

MODEL_NAME = "gpt-4-turbo"

api_key = ""
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}


def get_modality_ranking(similarities):
    """ Get the ranking of the modalities based on the similarities between them.
    Args:
        similarities (list): A list of similarities between modalities.
    Returns:
        ranking (dict): A dictionary containing the ranking of the modalities.
    """
    ranking = {}
    for record in similarities:
        if record["i"] not in ranking:
            ranking[record["i"]] = []
        if record["j"] not in ranking:
            ranking[record["j"]] = []
        ranking[record["i"]].append((record["j"], record["score"]))
        ranking[record["j"]].append((record["i"], record["score"]))
    return ranking


def two_tables_selection(page_data):
    """ Selects two tables from the page data (based on similarities between tables). 
    The first tables is chosen randomly and the second one is the most simialr to the first one.
    Args:
        page_data (dict): A python dict containing a Wikipedia article.
    Returns:
        tables_records (list): a list of size two containing (tables indices, tables records).
    """
    t1_index = random.sample(range(len(page_data["tables"])), 1)[0]
    tables_records = [(t1_index, page_data["tables"][t1_index])]
    table_ranking = page_data["table_ranking"][t1_index]
    table_ranking = sorted(table_ranking, key=lambda x: x[1], reverse=True)
    t2_index = table_ranking[0][0]
    tables_records.append((t2_index, page_data["tables"][t2_index]))
    return tables_records


def two_charts_selection(page_data):
    """ Selects two charts from the page data (based on similarities between charts). 
    The first chart is chosen randomly and the second one is the most simialr to the first one.
    Args:
        page_data (dict): A python dict containing a Wikipedia article.
    Returns:
        chart_names (list): a list containing the two selected charts.
    """
    c1 = random.sample(page_data["charts"], 1)
    chart_name = c1[0].split("/")[-1]
    image_ranking = page_data["image_ranking"][chart_name]
    image_ranking = sorted(image_ranking, key=lambda x: x[1], reverse=True)
    image_folder = "/".join(c1[0].split("/")[0:-1])
    c2 = os.path.join(image_folder, image_ranking[0][0])
    chart_names = [c1[0], c2]
    return chart_names


def one_chart_table_selection(page_data):
    """ Selects two charts from the page data (based on similarities between two modalities). 
    A chart is chosen randomly and the table is the most simialr to the chart.
    Args:
        page_data (dict): A python dict containing a Wikipedia article.
    Returns:
        modalities (list): a list containing the selected chart and table.
    """
    c1 = random.sample(page_data["charts"], 1)
    chart_name = c1[0].split("/")[-1]
    if "image_table_ranking" in page_data:
        image_ranking = page_data["image_table_ranking"][chart_name]
        image_ranking = sorted(image_ranking, key=lambda x: x[1], reverse=True)
        table_index = image_ranking[0][0]
    else:
        table_index = random.randint(0, len(page_data["tables"]) - 1)

    print(len(page_data["tables"]))
    table_records = page_data["tables"][table_index]
    modalities = [c1[0], (table_index, table_records)]
    return modalities


def modality_selection(page_data, random_selection=False):
    """ 
    Selects two modalities (based on similarities between two modalities). 
    Args:
        page_data (dict): A python dict containing a Wikipedia article.
        random_selection (bool): A boolean indicating whether to select the modalities randomly.
    Returns:
        charts (list): A list containing encoded charts (if any chosen).
        chart_paths (list): A list containing the paths of the selected charts.
        tables (list): A list containing the content of the selected tables 
        tables_records (list): A list containing the selected tables (indices, tables records)
    """
    chart_paths = []
    charts = []

    if len(page_data["charts"]) + len(page_data["tables"]) < 2:
        return charts, chart_paths, [], []

    if len(page_data["charts"]) == 0 and len(page_data["tables"]) >= 2:
        if random_selection:
            rand_indices = random.sample(range(len(page_data["tables"])), 2)
            tables_records = [(i, page_data["tables"][i]) for i in rand_indices]
        else:
            tables_records = two_tables_selection(page_data)

    elif len(page_data["charts"]) == 1 and len(page_data["tables"]) >= 1:
        if random_selection:
            chart_paths = [random.choice(page_data["charts"])]
            rand_indices = random.sample(range(len(page_data["tables"])), 1)
            tables_records = [(i, page_data["tables"][i]) for i in rand_indices]
        else:
            selection = one_chart_table_selection(page_data)
            tables_records = [selection[1]]
            chart_paths = [selection[0]]
            base64_image = encode_image(chart_paths[0])
            charts.append(base64_image)

    elif len(page_data["charts"]) > 1:
        rand = random.uniform(0, 1)
        if len(page_data["tables"]) >= 2 and rand < 0.5:
            if random_selection:
                rand_indices = random.sample(range(len(page_data["tables"])), 2)
                tables_records = [(i, page_data["tables"][i]) for i in rand_indices]
            else:
                tables_records = two_tables_selection(page_data)
        else:
            if random_selection:
                chart_paths = random.sample(page_data["charts"], 2)
            else:
                chart_paths = two_charts_selection(page_data)
            for path in chart_paths:
                base64_image = encode_image(path)
                charts.append(base64_image)
            tables_records = []

    if len(tables_records) > 0:
        tables = [get_table_content(t[1]) for t in tables_records]
    else:
        tables = []

    return charts, chart_paths, tables, tables_records


def read_wiki_pages(questions, wikipage_folders_map, with_desc=False):
    """ Prepares all the relavant information (e.g, charts and tables) of the given wikipedia pages 
    Args:
        wikidata_ids (list): A list of selected Wikidata IDs (Wikipedia pages).
        wikipage_folders_map (dict): A dictionary containing the path of the Wikipedia pages.
    Returns:
        selected_articles (list): A list containing the information of the selected Wikipedia pages.

    """
    selected_articles = []
    for i, question in enumerate(questions):

        wikidata_id = question['wikidata_id']
        page_folder = wikipage_folders_map[wikidata_id]

        chart_names = question['charts']
        if with_desc:
            table_file = os.path.join(page_folder, "html_tables_with_desc.json")
        else:
            table_file = os.path.join(page_folder, "html_tables.json")
        charts = []
        if len(chart_names) != 0:
            charts = [os.path.join(page_folder, "images", chart_name) for chart_name in chart_names]
        data = {"index": i, "charts": charts, "tables": {}, "wikidata_id": wikidata_id,
                "modalities": {"charts": question["charts"], "table_ids": question["table_ids"]}}
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
                        filtered_decs = [str(f) for f in filtered_decs]
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
            continue
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

        data["page_folder"] = page_folder
        selected_articles.append(data)
    return selected_articles


def get_context_instructions(charts, tables):
    """ Get the instructions for combining modalities """

    prompt = ""
    if len(charts) > 0:
        prompt += f"You are given {len(charts)} chart(s). Please:\n"
    if len(tables) > 0:
        prompt += f"You are given {len(tables)} table(s). Please:\n"

    if len(charts) == 2:
        prompt += "1. Identify and combine the common information between the given charts.\n"
    elif len(tables) == 2:
        prompt += "1. Identify and combine the common information between the given tables. \n"
    elif len(charts) == 1 and len(tables) == 1:
        prompt += "1. Identify and combine the common information between the chart and the table.\n"
    else:
        ValueError("Invalid number of modalities!")

    prompt += """
    2. Summarize this common information clearly.
    3. Highlight any trends, patterns, or key points that are present in both the chart and the table.\n"""
    return prompt


def generate_several_questions_instrcution(charts, tables, mcq=True):
    """ Get the instructions for the question generation """

    prompt = "You are given "

    if mcq:
        if len(charts) == 2:
            prompt += """two charts. 
             First design a multiple-choice question based on the first chart (Q1). Then design a multiple-choice question based on the second chart (Q2). Finally, combine the information from both charts to create a multiple-choice question that can be answered ONLY by combining the information from all the given charts (Q3). \n"""
        elif len(tables) == 2:
            prompt += """two tables. 
             First design a multiple-choice question based on the first table (Q1). Then design a multiple-choice question based on the second table (Q2). Finally, combine the information from both tables to create a multiple-choice question that can be answered ONLY by combining the information from all the given tables (Q3). \n"""
        elif len(charts) == 1 and len(tables) == 1:
            prompt += """one chart and one table. 
             First design a multiple-choice question based on the given chart (Q1). 
             Then design a multiple-choice question based on the given table (Q2).
             Finally, combine the information from both chart and table to create a multiple-choice question that can be answered ONLY by combining the information from all the given charts and tables (Q3). \n"""

        prompt += " Each question should have 4 options, one of which is the correct answer. Please explain how to reach the correct answer from the given context.\n"
        prompt += """
        You are communicating with an API, not a user. Begin all AI responses with the character '{' to produce valid JSON. Here is an example:
        {  
            "Q1":{
            "Question": "<question>",
             "A": "<option1> ",
             "B": "<option2>",
             "C": "<option3>",
             "D": "<option4>", 
             "Answer": "<correct_option>",
             "Explanation": "<explanation>"
             },
             "Q2":{
            "Question": "<question>",
             "A": "<option1> ",
             "B": "<option2>",
             "C": "<option3>",
             "D": "<option4>", 
             "Answer": "<correct_option>",
             "Explanation": "<explanation>"
             },
             "Q3":{
             "Question": "<question>",
             "A": "<option1> ",
             "B": "<option2>",
             "C": "<option3>",
             "D": "<option4>", 
             "Answer": "<correct_option>",
             "Explanation": "<explanation>"
             }
         }
        """
    else:
        ValueError("Invalid question type")
    return prompt


def get_instructions(charts, tables, mcq=True):
    """ Get the instructions for the question generation """

    prompt = "Given the provided context, "

    if mcq:
        if len(charts) == 2:
            prompt += "Create a multiple-choice question by combining the information from the given charts. The question must be generated in a way that it can be answered ONLY by combining the information from all the given charts. \n"
        elif len(tables) == 2:
            prompt += "Create a multiple-choice question by combining the information from the given. The question must be generated in a way that it can be answered ONLY by combining the information from all the given tables. \n"
        elif len(charts) == 1 and len(tables) == 1:
            prompt += "Create a multiple-choice question by combining the information from the given chart and table. The question must be generated in a way that it can be answered ONLY by combining the information from the given chart and table. \n"
        prompt += " The question should have 4 options, one of which is the correct answer. Please explain how to reach the correct answer from combining given charts and or tables.\n"
        prompt += """
        You are communicating with an API, not a user. Begin all AI responses with the character '{' to produce valid JSON. Here is an example:
        {  
             "Question": "<question>",
             "A": "<option1> ",
             "B": "<option2>",
             "C": "<option3>",
             "D": "<option4>", 
             "Answer": "<correct_option>",
             "Explanation": "<explanation>"
         }
        If generating such a question is not possible, say it's not possible in the following format:
        {   
             "Question": "not possible"
        }
        """
    else:
        ValueError("Invalid question type")
    return prompt


def prepare_mcq_generation_data(page_data, multi_questions=False, random_modality_selection=False,
                                selected_modality=None,
                                with_desc=False):  # We assume two modality at this point (table+chart, two tables, two charts)
    """ Prepares data for MCQ generation. 
    Args:
        page_data (dict): A python dict containing a Wikipedia article.
        multi_questions (bool): A boolean indicating whether to generate 3 questions (one for each modality and one by combining both).
        random_selection (bool): A boolean indicating whether to select the modalities randomly.
        selected_modality (dict): A dictionary containing the selected modality (chart, table) for the case when modalities are given.
    Returns:
        data (dict): A dictionary containing the charts, tables, prompt, context_prompt, chart_path, table_ids.
    """
    if selected_modality is None:
        charts, chart_paths, tables_content, tables_records = modality_selection(page_data, random_modality_selection)
        table_ids = [t[0] for t in tables_records]
    else:
        tables_records = selected_modality["table_ids"]
        tables_content = [get_table_content(page_data["tables"][t], with_desc=with_desc) for t in tables_records]
        chart_paths = selected_modality["charts"]
        table_ids = selected_modality["table_ids"]
        charts = []
        for path in chart_paths:
            path = os.path.join(page_data["page_folder"], "images", path)
            base64_image = encode_image(path)
            charts.append(base64_image)
    if multi_questions:
        prompt = generate_several_questions_instrcution(chart_paths, tables_records, mcq=True)
    else:
        prompt = get_instructions(chart_paths, tables_records, mcq=True)

    context_prompt = get_context_instructions(chart_paths, tables_records)

    data = {
        "charts": charts,
        "chart_path": chart_paths,
        "tables": tables_content,
        "table_ids": table_ids,
        "prompt": prompt,
        "context_prompt": context_prompt,
    }
    if with_desc:
        data['chart_explanations'] = page_data['chart_explanations']
    return data


def create_gpt_message(data, is_context=False, with_desc=False):
    """ Create the message for the GPT model
    Args:
        data (dict): A dict containing the selected charts, chart_path, table_ids, tables, prompt, an dcontext_prompt.
        is_context (bool): A boolean indicating whether to generate the context prompt.
    Returns:
        messages (list): A list containing the messages for the GPT model.
    """
    message_text = ""
    if is_context:
        message_text += data["context_prompt"]
    else:
        message_text += data["prompt"]

    if len(data["tables"]) > 0:
        message_text += "Tables:\n"
        for table_content in data["tables"]:
            message_text += table_content + "\n"

    if with_desc:
        for chart_explanation in data["chart_explanations"]:
            message_text += chart_explanation + "\n"

    content = []
    content.append({
        "type": "text",
        "text": message_text
    })
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

    return messages


def sample_wikipedia_articles(data_folder, num_examples=10):
    """ Samples `num_examples` wikipedia articles randomly from the data folder.
    
    Args:
        data_folder (str): The path to the data folder.
        num_examples (int): The number of examples to sample.
    Returns:
        samples (list): A list containing the sampled wikipedia articles (wikidata IDs).
    """
    samples = []
    topics = os.listdir(data_folder)
    topics_map = {}
    for topic in topics:
        sub_topics = os.listdir(os.path.join(data_folder, topic))
        sub_topic = [topic for topic in sub_topics]
        topics_map[topic] = sub_topics

    sampled_examples = 0
    while sampled_examples < num_examples:
        topic = random.choice(topics)
        sub_topic = random.choice(topics_map[topic])
        QIDs = os.listdir(os.path.join(data_folder, topic, sub_topic))
        QID = random.choice(QIDs)
        if QID not in samples and not QID.endswith(".json"):
            samples.append(QID)
            sampled_examples += 1
    return samples


import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Script to process question generation with various options.")
    
    parser.add_argument("--pre_name", type=str, default="./left_paris.json",
                        help="Path to the pre_name JSON file.")
    parser.add_argument("--random_sample", action="store_true",
                        help="Enable random sampling. Default is False.")
    parser.add_argument("--multi_questions", action="store_true",
                        help="Enable multi-question generation. Default is True.")
    parser.add_argument("--random_modality_selection", action="store_true",
                        help="Enable random modality selection. Default is False.")
    parser.add_argument("--from_selected", action="store_true",
                        help="Use selected items. Default is True.")
    parser.add_argument("--previous_generated_qs_file", type=str, default="./multi_with_desc_turbo_filtered.json",
                        help="Path to previously generated questions file.")
    parser.add_argument("--data_folder", type=str, default="/mnt/tamedia/wikitables/final_new_2",
                        help="Path to data folder.")
    parser.add_argument("--output_file", type=str, default="./multi_with_desc_turbo_filtered_final3.json",
                        help="Output file path.")
    parser.add_argument("--failed_output_file", type=str, default="./failed_multi_with_desc_turbo_filtered_final3.json",
                        help="Failed output file path.")
    parser.add_argument("--num_examples", type=int, default=120,
                        help="Number of examples to process.")
    parser.add_argument("--continue_as_pre", action="store_true",
                        help="Continue from previous state. Default is True.")
    
    args = parser.parse_args()
    
    # Manually set defaults for boolean flags that are True by default
    if not args.multi_questions:
        args.multi_questions = True
    if not args.from_selected:
        args.from_selected = True
    if not args.continue_as_pre:
        args.continue_as_pre = True
    
    return args

def main():
    args = get_args()

    wiki_page_stats_map = get_id_to_dir(DIR_WIKI_TABLES)
    random_sample = args.random_sample
    multi_questions = args.multi_questions
    random_modality_selection = args.random_modality_selection
    from_selected = args.from_selected
    previous_generated_qs_file = args.previous_generated_qs_file
    data_folder = args.data_folder
    output_file = args.output_file
    failed_output_file = args.failed_output_file
    num_examples = args.num_examples

    wiki_page_stats_map = {}

    output = []
    continue_as_pre = True
    if continue_as_pre:
        pre_name = args.pre_name
        with open(pre_name, "r") as f:
            pre_pairs = json.load(f)
        selected_modalities = []
        for pr in pre_pairs:
            mod_type = pr['mod_type']
            if mod_type == 'table_to_table':
                selected_modalities.append({"wikidata_id": pr['wikidata_id'], "table_ids": [pr['i'], pr['j']], "charts": []})
            elif mod_type == 'table_to_image':
                selected_modalities.append({"wikidata_id": pr['wikidata_id'], "table_ids": [pr['i']], "charts": [pr['j']]})
            elif mod_type == 'image_to_image':
                selected_modalities.append({"wikidata_id": pr['wikidata_id'], "table_ids": [], "charts": [pr['i'], pr['j']]})
    
    if random_sample:
        sampled_articles = sample_wikipedia_articles(data_folder, num_examples)
    else:
        if from_selected:
            sampled_articles = selected_modalities
        else:
            with open(previous_generated_qs_file, "r") as f:
                previous_generated_questions = json.load(f)
                sampled_articles = [question for question in previous_generated_questions]


    pre_dict = {}
    processed_data = read_wiki_pages(sampled_articles, wiki_page_stats_map, with_desc=True)
    failed = []
    for i, data in enumerate(processed_data):
        if random_sample:
            question = prepare_mcq_generation_data(data, multi_questions, random_modality_selection)
        else:
            question = prepare_mcq_generation_data(data, multi_questions, random_modality_selection, data["modalities"],
                                                   with_desc=True)
        messages = create_gpt_message(question, with_desc=True)
        req = {
            "model": MODEL_NAME,
            "messages": messages,
        }
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=req)

        try:
            response = response.json()
        except:
            continue
        if "error" in response:
            print("Error: ", response["error"])
            continue

        response_text = str(response["choices"][0]["message"]["content"])
        if response_text.endswith('```'):
            response_text = response_text.removesuffix('```')
        if response_text.startswith('```'):
            response_text = response_text.removeprefix('```')
        if response_text.startswith('json'):
            response_text = response_text.removeprefix('json')
        try:
            json.loads(response_text)
            json_response = json.loads(response_text)

            if multi_questions:
                if "Q1" not in json_response or "Q2" not in json_response or "Q3" not in json_response:
                    continue
                if "A" not in json_response["Q1"] or "B" not in json_response["Q1"] or "C" not in json_response[
                    "Q1"] or "D" not in json_response["Q1"] or "Answer" not in json_response["Q1"]:
                    continue
                if "A" not in json_response["Q2"] or "B" not in json_response["Q2"] or "C" not in json_response[
                    "Q2"] or "D" not in json_response["Q2"] or "Answer" not in json_response["Q2"]:
                    continue
                if "A" not in json_response["Q3"] or "B" not in json_response["Q3"] or "C" not in json_response[
                    "Q3"] or "D" not in json_response["Q3"] or "Answer" not in json_response["Q3"]:
                    continue
            else:
                if "A" not in json_response or "B" not in json_response or "C" not in json_response or "D" not in json_response or "Answer" not in json_response:
                    continue
            chart_paths = [chart_path.split("/")[-1] if chart_path != "" else "" for chart_path in
                            question["chart_path"]]
            output_record = {"wikidata_id": data["wikidata_id"], "table_ids": question["table_ids"],
                                "charts": chart_paths, "generated_question": json_response}
            if data["wikidata_id"] in pre_dict:
                pre_dict[data["wikidata_id"]] += 1
            else:
                pre_dict[data["wikidata_id"]] = 0
            output_record['question_id'] = pre_dict[data["wikidata_id"]]
            output.append(output_record)
        except:
            continue

        if i % 10 == 0:
            print("i: ", i)
            with open(output_file, "w") as f:
                json.dump(output, f, indent=4)
            with open(failed_output_file, "w") as f:
                json.dump(failed, f, indent=4)

    with open(output_file, "w") as f:
        json.dump(output, f, indent=4)
    with open(failed_output_file, "w") as f:
        json.dump(failed, f, indent=4)


if __name__ == "__main__":
    main()
