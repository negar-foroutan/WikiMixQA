from FlagEmbedding import FlagReranker
from argparse import ArgumentParser

import json
import os
import itertools


categories = json.load(open("data/category_taxonomy.json"))

def get_image_descriptions(filename):
    descriptions = []
    with open(filename) as f:
        for line in f:
            data = json.loads(line)
            try:
                output = json.loads(
                    data["choices"][0]["message"]["content"], strict=False
                )
                if output["chart"] == "yes" and output["data"] is not None:
                    if isinstance(
                        output["data"], dict
                    ):  # for some reason the output is a dict
                        output["data"] = json.dumps(output["data"])
                    descriptions.append((data["filename"], output["data"]))
            except json.decoder.JSONDecodeError:
                print(data["choices"][0]["message"]["content"])
            except KeyError:
                print(data)
    return descriptions


def get_table_descriptions(filename, field="desc"):
    descriptions = []
    with open(filename) as f:
        for i, line in enumerate(f):
            table = json.loads(line)
            descriptions.append((i, table[field]))
    return descriptions


def get_combinations(descriptions):
    # table-to-table embeddings
    combinations = list(itertools.combinations(descriptions, 2))
    pairs = [[q1[1], q2[1]] for q1, q2 in combinations]
    indices = [[q1[0], q2[0]] for q1, q2 in combinations]
    return pairs, indices


def compute_scores(
    pairs,
    indices,
    reranker,
    output_filname,
):
    scores = reranker.compute_score(pairs, normalize=True)
    # put scores back to a list if we have a single score
    if not isinstance(scores, list):
        scores = [scores]
    with open(output_filname, "w") as f:
        for (i, j), score in zip(indices, scores):
            f.write(json.dumps({"i": i, "j": j, "score": score}) + "\n")


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--input-dir", type=str, required=True)
    argparser.add_argument(
        "--model-name", type=str, required=True, default="BAAI/bge-reranker-v2-m3"
    )
    argparser.add_argument("--use-fp16", action="store_true", default=False)
    argparser.add_argument("--device", type=str, default="cuda:0")
    args = argparser.parse_args()

    reranker = FlagReranker(
        args.model_name, use_fp16=args.use_fp16, device=args.device
    )  # Setting use_fp16 to True speeds up computation with a slight performance degradation

    # loop over all tables
    for topic, v in categories.items():
        topic_dir = os.path.join(args.input_dir, topic)
        for subtopic in v.keys():
            subtopic_dir = os.path.join(topic_dir, subtopic)
            for qid in os.listdir(subtopic_dir):
                if not os.path.isdir(os.path.join(subtopic_dir, qid)):
                    continue
                print(f"Processing {subtopic}/{qid}")
                tables_filename = os.path.join(
                    subtopic_dir, qid, "html_tables_with_desc.json"
                )
                images_filename = os.path.join(subtopic_dir, qid, "gpt-4-turbo.json")
                qid_with_tables = False
                if os.path.exists(tables_filename):
                    table_descriptions = get_table_descriptions(tables_filename)
                    output_filename = os.path.join(
                        subtopic_dir,
                        qid,
                        f"table_to_table_{os.path.basename(args.model_name)}.json",
                    )
                    if not os.path.exists(output_filename):
                        if len(table_descriptions) > 1:
                            # get all combinations of tables
                            table_pairs, table_indices = get_combinations(
                                table_descriptions
                            )
                            # table-to-table embeddings
                            compute_scores(
                                table_pairs, table_indices, reranker, output_filename
                            )
                    qid_with_tables = True
                if os.path.exists(images_filename):
                    image_descriptions = get_image_descriptions(images_filename)
                    output_filename = os.path.join(
                        subtopic_dir,
                        qid,
                        f"image_to_image_{os.path.basename(args.model_name)}.json",
                    )
                    if not os.path.exists(output_filename):
                        if len(image_descriptions) > 1:
                            # get all combinations of tables
                            image_pairs, images_filenames = get_combinations(
                                image_descriptions
                            )
                            # image-to-image embeddings
                            compute_scores(
                                image_pairs, images_filenames, reranker, output_filename
                            )
                    if (
                        qid_with_tables
                        and len(table_descriptions) > 0
                        and len(image_descriptions) > 0
                    ):
                        output_filename = os.path.join(
                            subtopic_dir,
                            qid,
                            f"table_to_image_{os.path.basename(args.model_name)}.json",
                        )
                        if not os.path.exists(output_filename):
                            combinations, indices = [], []
                            for table in table_descriptions:
                                for image in image_descriptions:
                                    combinations.append([table[1], image[1]])
                                    indices.append((table[0], image[0]))
                            # table-to-image embeddings
                            compute_scores(
                                combinations,
                                indices,
                                reranker,
                                output_filename,
                            )
