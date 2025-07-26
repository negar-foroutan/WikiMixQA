import json
import random
import os
from weasyprint import HTML

from tools.interpretation import remove_previous, read_qs

qs_dict = {}
name_list = []

def generation_to_pdf(json_list, id_to_dir, is_multi=False, path_to_save='./'):
    explanation_keys = ['Explanation']

    for js in json_list:
        id_ = js['wikidata_id']
        if id_ in id_to_dir:
            dir_ = f'{id_to_dir[id_]}/html_tables.json'
            html_list = read_jsonl(dir_)
            charts = js['charts']
            charts_path = f'{id_to_dir[id_]}/images'
            chart_html = [f"<img src=\"{charts_path}/{chart}\" alt=\"{chart}\">" for chart in charts]
            htmls = []
            ch = False
            for table_ind in js["table_ids"]:
                try:
                    htmls.append(html_list[table_ind]['html'])
                except Exception as e:
                    print('error: ', id_, table_ind, id_to_dir[id_], e)
                    ch = True
                    break
            if ch:
                continue
            questions = [js['generated_question']] if not is_multi else js['generated_question'].values()
            questions_html = []
            try:
                for i, q in enumerate(questions):
                    explanation_key = None
                    for key in explanation_keys:
                        if key in q.keys():
                            explanation_key = key
                            break
                    if explanation_key is None:
                        explanation_key = [k for k in q.keys() if k in ['A', 'B', 'C', 'D', 'Answer', 'Question']][0]
                        explanation_keys.append(explanation_key)

                    questions_html.append(f"""
            <div class="question">
                <h2>{str(i+1)}. {q['Question']}</h2>
                <div class="answers">
                    <p>A: {q['A']}</p>
                    <p>B: {q['B']}</p>
                    <p>C: {q['C']}</p>
                    <p>D: {q['D']}</p>
                </div>
                <div class="explanation">
                    Answer: {q['Answer']} - {q[explanation_key]}
                </div>
            </div>
                """
                                        )
            except Exception as e:
                print('error: ', id_, id_to_dir[id_], e)
                continue
            html_content = """
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>Multiple Tables and Charts to PDF</title>
                <style>
                    table {
                        width: 100%;
                        font-size: 10px; /* Reduces the font size throughout the table */
                        border-collapse: fixed;
                        word-wrap: break-word;
                        margin-bottom: 20px; /* Space between tables */
                        page-break-inside: avoid;
                        overflow-x: auto;
                    }
                    th, td {
                        border: 1px solid #ddd; /* Adds a subtle border to each cell */
                        padding: 4px;
                        page-break-inside: avoid;
                        text-align: left;
                    }
                    thead {
                        background-color: #f4f4f4; /* Light gray background for header */
                        font-weight: bold; /* Makes header text bold */
                    }
                    
                    tfoot {
                        font-weight: bold;
                        background-color: #f4f4f4;
                    }
                    thead {
                        background-color: #f4f4f4;
                        font-weight: bold;
                    }
                    caption {
                        background-color: #f2f2f2;
                        margin-bottom: -1px;
                        border: 1px solid #aaa;
                        padding: 0.2em 0.4em;
                    }
                    img {
                        max-width: 100%;
                        height: auto;
                        display: block;
                        page-break-inside: avoid;
                        margin-bottom: 20px;
                    }
                </style>
            </head>
            <body>
                """ + """
                """.join(htmls) + """
                """ + """
                """.join(chart_html) + """
                """ + """
                """.join(questions_html) + """
            </body>
            </html>
            """
            if id_ in qs_dict.keys():
                qs_dict[id_] = qs_dict[id_] + 1
            else:
                qs_dict[id_] = 0
            os.makedirs(path_to_save, exist_ok=True)
            HTML(string=html_content, base_url=charts_path).write_pdf(f'{path_to_save}/{id_}_{qs_dict[id_]}.pdf')
            name_list.append(f'{id_}_{qs_dict[id_]}')

        else:
            print(f"ID {id_} not found")


def read_jsonl(filename, is_list=False):
    if not is_list:
        with open(os.path.join(filename), 'r') as f:
            jsonl = f.readlines()
        jsonl = [json.loads(j) for j in jsonl]
    else:
        with open(os.path.join(filename), 'r') as f:
            jsonl = json.load(f)

    return jsonl


def correct_question_id_without_path(q_list, out_file=None):
    q_dict = {}
    for q in q_list:
        wiki_id = q['wikidata_id']
        if wiki_id in q_dict:
            q_dict[wiki_id] += 1
        else:
            q_dict[wiki_id] = 0
        q['question_id'] = q_dict[wiki_id]
    rev_ind_dict = {}
    for q_id, num in q_dict.items():
        if num in rev_ind_dict:
            rev_ind_dict[num] += 1
        else:
            rev_ind_dict[num] = 1
    for num, num_num in sorted(rev_ind_dict.items()):
        print(f"{num + 1}:", num_num)
    with open(out_file, "w") as f:
        json.dump(q_list, f, indent=4)

def correct_question_id(in_files, out_file=None):
    q_list = []
    for in_file in in_files:
        with open(in_file, "r") as f:
            previous_generated_questions = json.load(f)
            q_list += [question for question in previous_generated_questions]
    if out_file is None:
        out_file = in_files[0]
    correct_question_id_without_path(q_list, out_file=out_file)



def equal_mods(mod1, mod2):
    for i in range(len(mod1)):
        if str(mod1[i]) != str(mod2[i]):
            return False
    return True


def correct_generation_with_pre(q_path, current_pairs_path, pre_pairs_path):
    with open(current_pairs_path, "r") as f:
         current_pairs = json.load(f)
    with open(pre_pairs_path, "r") as f:
        pre_pairs = json.load(f)

    q_mods = []
    with open(q_path, "r") as f:
        previous_generated_questions = json.load(f)
        for q in previous_generated_questions:
            ls = []
            ls += [str(t) for t in q['table_ids']]
            ls += [c for c in q['charts']]
            q_mods.append((q, ls))
    print(len(q_mods))
    pre_pair_gen = []
    for topic, mod_prs in pre_pairs.items():
        for mod, prs in mod_prs.items():
            current_pairs[topic][mod] = remove_previous(current_pairs[topic][mod], prs)
            for pr in prs:
                filtered = []
                for q in q_mods:
                    if not equal_mods([str(pr['i']), str(pr['j'])], q[1]):
                        filtered.append(q)
                    else:
                        pre_pair_gen.append(q)
                q_mods = filtered
    print(len(q_mods))
    left_pairs = []
    for topic, mod_prs in current_pairs.items():
        for mod, prs in mod_prs.items():
            for pr in prs:
                cnt = 0
                filtered = []
                ch = False
                for q in q_mods:
                    if equal_mods([str(pr['i']), str(pr['j'])], q[1]):
                        ch = True
                        cnt += 1
                        if cnt > 1:
                            filtered.append(q)
                if not ch:
                    pr['mod_type'] = mod
                    left_pairs.append(pr)
                for duplicate in filtered:
                    q_mods.remove(duplicate)
    total = 0
    for topic, prs in current_pairs.items():
        for mod, pr in prs.items():
            total += len(pr)
    print(len(q_mods), total, len(left_pairs))

    q_list = [q[0] for q in q_mods]
    output_file = '../generation/cleared_gen.json'
    with open(output_file, "w") as f:
        json.dump(q_list, f, indent=4)
    left_path = '../generation/left_paris.json'
    with open(left_path, "w") as f:
        json.dump(left_pairs, f)
    pre_path = '../generation/pre_pair_gen.json'
    with open(pre_path, "w") as f:
        json.dump(pre_pair_gen, f, indent=4)
    print(len(pre_pair_gen))


def make_full_dataset(paths):
    qs_list = [read_qs(path) for path in paths]
    corrs = []
    wrongs = []
    total = 0
    final_questions = []
    corr_len = 0
    for qs in qs_list:
        total += len(qs)
        yes_no_dict = {'yes.': True, 'yes': True, 'no.': False, 'no': False, 'yes,': True, 'no,': False}
        corr = []
        wrong = []
        for q in qs:
            try:
                response = yes_no_dict[q["InternVL2-evaluator"]["enough-information"].lower()[:3]]
                if response:
                    corr.append(q)
                else:
                    wrong.append(q)
            except Exception as e:
                wrong.append(q)
                continue
        corr_len += len(corr)
        corrs.append(corr)
        wrongs.append(wrong)
    q_num_left = 2000 - corr_len
    for i in range(len(corrs)):
        final_questions += corrs[i]
        sample_num = int(q_num_left * len(qs_list[i]) / total) + 1
        random.shuffle(wrongs[i])
        final_questions += wrongs[i][:sample_num]
        print(len(corrs[i]), sample_num, paths[i])
    print(len(final_questions))
    correct_question_id_without_path(final_questions, out_file='./final_data.json')






