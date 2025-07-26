import json
import os

import numpy as np
import pandas as pd
from pathlib import Path

from matplotlib import pyplot as plt
from numpy import sort

from config.config import DIR_WIKI_TABLES
from tools.utils import get_id_to_dir, get_subject_to_dir, get_subsub_to_dir


def get_accuracy(di):
    ls = ['Q1_answer_correctness', 'Q2_answer_correctness', 'Q3_answer_correctness']
    count = {q: {"correct": 0, "wrong": 0, "impossible": 0, "neutral": 0} for q in ls}
    for k, v in di.items():
        for q in ls:
            try:
                if v[q] == 'correct':
                    count[q]["correct"] += 1
                elif v[q] == 'wrong':
                    count[q]["wrong"] += 1
                elif v[q] == 'impossible':
                    count[q]["impossible"] += 1
                elif v[q] == 'neutral':
                    count[q]["neutral"] += 1
            except:
                continue

    for q in ls:
        total = 0
        for k, v in count[q].items():
            total += v
        try:
            for k, v in count[q].items():
                count[q][k] = int(v / total * 100)
        except:
            pass
        count[q]["total"] = total

    # pretty print
    print('Accuracy')
    for q in ls:
        print(f'{q}: {count[q]}')


def get_check_sums_accuracy(di, req):
    ls = ['_tables_required_to_answer', '_charts_required_to_answer']
    count = {q: {"True": 0, "False": 0} for q in req.keys()}

    for k, v in di.items():
        for q, r in req.items():
            try:
                val = v[f'{q}_tables_required_to_answer'] + v[f'{q}_charts_required_to_answer']
                if not val >= 0:
                    continue
                if v[f'{q}_tables_required_to_answer'] + v[f'{q}_charts_required_to_answer'] == r:
                    count[q]["True"] += 1
                else:
                    count[q]["False"] += 1
            except:
                continue

    for q in req.keys():
        total = 0
        for k, v in count[q].items():
            total += v

        try:
            for k, v in count[q].items():
                count[q][k] = int(v / total * 100)
        except:
            pass
        count[q]["total"] = total

    # pretty print
    print('Check sums accuracy')
    for q in req.keys():
        print(f'{q}: {count[q]}')


def information_acc(di):
    ls = ['Q1_enough_information_to_answer', 'Q2_enough_information_to_answer', 'Q3_enough_information_to_answer']
    count = {q: {"yes": 0, "no": 0} for q in ls}
    for k, v in di.items():
        for q in ls:
            try:
                if v[q] == 'yes':
                    count[q]["yes"] += 1
                elif v[q] == 'no':
                    count[q]["no"] += 1
            except:
                continue

    for q in ls:
        total = 0
        for k, v in count[q].items():
            total += v
        try:
            for k, v in count[q].items():
                count[q][k] = int(v / total * 100)
        except:
            pass
        count[q]["total"] = total

    # pretty print
    print('Information accuracy')
    for q in ls:
        print(f'{q}: {count[q]}')


def get_check_sums_accuracy_with_info(di, req):
    ls = ['_tables_required_to_answer', '_charts_required_to_answer']
    count = {q: {"tp": 0, "fp": 0, "tn": 0, "fn": 0} for q in req.keys()}

    for k, v in di.items():
        for q, r in req.items():
            try:
                val = v[f'{q}_tables_required_to_answer'] + v[f'{q}_charts_required_to_answer']
                if not val >= 0:
                    continue
                if v[f'{q}_tables_required_to_answer'] + v[f'{q}_charts_required_to_answer'] == r and v[
                    f'{q}_enough_information_to_answer'] == 'yes':
                    count[q]["tp"] += 1
                elif v[f'{q}_tables_required_to_answer'] + v[f'{q}_charts_required_to_answer'] == r and v[
                    f'{q}_enough_information_to_answer'] == 'no':
                    count[q]["fp"] += 1
                elif v[f'{q}_tables_required_to_answer'] + v[f'{q}_charts_required_to_answer'] != r and v[
                    f'{q}_enough_information_to_answer'] == 'yes':
                    count[q]["fn"] += 1
                elif v[f'{q}_tables_required_to_answer'] + v[f'{q}_charts_required_to_answer'] != r and v[
                    f'{q}_enough_information_to_answer'] == 'no':
                    count[q]["tn"] += 1
            except:
                continue

    for q in req.keys():
        total = 0
        for k, v in count[q].items():
            total += v
        try:
            for k, v in count[q].items():
                count[q][k] = int(v / total * 100)
        except:
            pass
        count[q]["total"] = total

    # pretty print
    print('Check sums accuracy with information')
    for q in req.keys():
        print(f'{q}: {count[q]}')


def read_and_aggregate_csvs(directory, file_pattern):
    csv_files = Path(directory).glob(file_pattern)
    dfs = [pd.read_excel(file) for file in csv_files]

    combined_df = pd.concat(dfs, ignore_index=True)

    column_names_list = combined_df.columns.tolist()
    qs = {wiki_id[:-2]: {} for wiki_id in combined_df['wikidata_id'].unique()}

    for col in column_names_list:
        if col == 'wikidata_id':
            continue
        aggregated_col = combined_df.groupby('wikidata_id')[col].apply(
            lambda x: x.mode().iloc[0] if not x.mode().empty else None).reset_index()
        for i, row in aggregated_col.iterrows():
            qs[row['wikidata_id'][:-2]][col] = row[col]
    return qs


def get_similarity_generated_questions(q_jsons, id_to_dir):
    scores = {}
    for q in q_jsons:
        charts = q['charts']
        table_ids = q['table_ids']
        table_ids = [str(t) for t in table_ids]
        if len(table_ids) == 2:
            similarity_path = f'{id_to_dir[q["wikidata_id"]]}/table_to_table_bge-reranker-v2-m3.json'
        elif len(charts) == 2:
            similarity_path = f'{id_to_dir[q["wikidata_id"]]}/image_to_image_bge-reranker-v2-m3.json'
        else:
            similarity_path = f'{id_to_dir[q["wikidata_id"]]}/table_to_image_bge-reranker-v2-m3.json'
        modalities = table_ids + charts
        with (open(similarity_path, "r") as f):
            for row_json in f.readlines():
                row = json.loads(row_json)
                if (str(row['i']) == modalities[0] and str(row['j']) == modalities[1]) or (
                        str(row['j']) == modalities[0] and str(row['i']) == modalities[1]):
                    scores[q['wikidata_id']] = row['score']
                    break
    return scores


def get_similarity_correlation(q_eval, q_jsons, q_number):
    id_to_dir = get_id_to_dir(DIR_WIKI_TABLES)
    scores = get_similarity_generated_questions(q_jsons, id_to_dir)
    q_key = f'Q{q_number}'

    correctness_arr = []
    modalities_arr = []
    information_arr = []
    similarities_arr = []
    for q_id, q in q_eval.items():
        try:
            correctness_check = 1 if q[f'{q_key}_answer_correctness'] == 'correct' else 0
            modalities_check = 1 if q[f'{q_key}_tables_required_to_answer'] + q[
                f'{q_key}_charts_required_to_answer'] == 2 else 0
            information_check = 1 if q[f'{q_key}_enough_information_to_answer'] == 'yes' else 0
            similarities_arr.append(scores[q_id])
            correctness_arr.append(correctness_check)
            modalities_arr.append(modalities_check)
            information_arr.append(information_check)
        except Exception as e:
            continue
    correctness_arr = np.array(correctness_arr)
    modalities_arr = np.array(modalities_arr)
    information_arr = np.array(information_arr)
    similarities_arr = np.array(similarities_arr)

    # computing the correlation with similarities
    corr_correctness = np.corrcoef(correctness_arr, similarities_arr)[0, 1]
    corr_modalities = np.corrcoef(modalities_arr, similarities_arr)[0, 1]
    corr_information = np.corrcoef(information_arr, similarities_arr)[0, 1]
    print("---------------------------------")
    print(f'Correlation with similarities')
    print(f'Correctness check: {corr_correctness}')
    print(f'Modalities check: {corr_modalities}')
    print(f'Information check: {corr_information}')
    print("---------------------------------")


def get_statistics():
    folder = './zero_with_desc'
    print(folder)
    result_df = read_and_aggregate_csvs(folder, '*.xlsx')
    print('---------------------')
    get_accuracy(result_df)
    print()
    req = {'Q1': 2, 'Q2': 2, 'Q3': 2}
    get_check_sums_accuracy(result_df, req)
    print()
    information_acc(result_df)
    print()
    get_check_sums_accuracy_with_info(result_df, req)
    print('---------------------')
    print()

    folder = './multi_with_desc'
    print(folder)
    result_df = read_and_aggregate_csvs(folder, '*.xlsx')
    print('---------------------')
    get_accuracy(result_df)
    print()
    req = {'Q1': 1, 'Q2': 1, 'Q3': 2}
    get_check_sums_accuracy(result_df, req)
    print()
    information_acc(result_df)
    print()
    get_check_sums_accuracy_with_info(result_df, req)
    print('---------------------')
    print()

    folder = './zero'
    print(folder)
    result_df = read_and_aggregate_csvs(folder, '*.xlsx')
    print('---------------------')
    get_accuracy(result_df)
    print()
    req = {'Q1': 2, 'Q2': 2, 'Q3': 2}
    get_check_sums_accuracy(result_df, req)
    print()
    information_acc(result_df)
    print()
    get_check_sums_accuracy_with_info(result_df, req)
    print('---------------------')
    print()

    folder = './multi'
    print(folder)
    result_df = read_and_aggregate_csvs(folder, '*.xlsx')
    print('---------------------')
    get_accuracy(result_df)
    print()
    req = {'Q1': 1, 'Q2': 1, 'Q3': 2}
    get_check_sums_accuracy(result_df, req)
    print()
    information_acc(result_df)
    print()
    get_check_sums_accuracy_with_info(result_df, req)
    print('---------------------')
    print()


def get_similarity_by_dir(dir_dict, r_level=0):
    dir_dict = {k: [v] for k, v in dir_dict.items()}
    for i in range(r_level):
        dir_dict_tmp = {}
        for k, v in dir_dict.items():
            dir_dict_tmp[k] = []
            for directory in v:
                for sub_dir in os.listdir(directory):
                    dir_dict_tmp[k].append(f"{directory}/{sub_dir}")
        dir_dict = dir_dict_tmp

    scores_dict = {}
    suffix = '_bge-reranker-v2-m3.json'
    for k, v in dir_dict.items():

        modal_dict = {'table_to_table': [], 'table_to_image': [],
                      'image_to_image': [], 'total': []}
        for path in v:
            for similarity_type, similarity_scores in modal_dict.items():
                similarity_path = f'{path}/{similarity_type}{suffix}'
                try:
                    with (open(similarity_path, "r") as f):
                        for row_json in f.readlines():
                            row = json.loads(row_json)
                            similarity_scores.append(row['score'])
                except Exception as e:
                    pass
        total_scores = []
        for similarity_type, similarity_scores in modal_dict.items():
            total_scores += similarity_scores
        modal_dict['total'] = total_scores
        scores_dict[k] = modal_dict

    return scores_dict

def get_similarity_full_by_dir(dir_dict, r_level=0):
    dir_dict = {k: [v] for k, v in dir_dict.items()}
    for i in range(r_level):
        dir_dict_tmp = {}
        for k, v in dir_dict.items():
            dir_dict_tmp[k] = []
            for directory in v:
                for sub_dir in os.listdir(directory):
                    dir_dict_tmp[k].append(f"{directory}/{sub_dir}")
        dir_dict = dir_dict_tmp

    scores_dict = {}
    total_table_dict = {}
    suffix = '_bge-reranker-v2-m3.json'
    for k, v in dir_dict.items():

        modal_dict = {'table_to_table': [], 'table_to_image': [],
                      'image_to_image': [], 'total': []}
        for path in v:
            path_id = path.split('/')[-1]
            for similarity_type, similarity_scores in modal_dict.items():
                similarity_path = f'{path}/{similarity_type}{suffix}'
                try:
                    with (open(similarity_path, "r") as f):
                        for row_json in f.readlines():
                            row = json.loads(row_json)
                            row['wikidata_id'] = path_id
                            similarity_scores.append(row)
                except Exception as e:
                    pass
            table_len_dict = {}
            table_path = f'{path}/tables.jsonl'
            with (open(table_path, "r") as f):
                for row_json in f.readlines():
                    row = json.loads(row_json)
                    table_len_dict[row['index']] = len(row['html'])
            total_table_dict[path_id] = table_len_dict


        total_scores = []
        for similarity_type, similarity_scores in modal_dict.items():
            total_scores += similarity_scores
        modal_dict['total'] = total_scores
        scores_dict[k] = modal_dict
    return scores_dict, total_table_dict



def get_similarity_mean_std(dir_dict, r_level=0):
    similarity_dict = get_similarity_by_dir(dir_dict, r_level)
    mean_std_dict = {k: {} for k in dir_dict.keys()}
    for k, modal_dict in similarity_dict.items():
        for similarity_type, similarity_scores in modal_dict.items():
            mean_std_dict[k][similarity_type] = (
            round(np.mean(similarity_scores), 2), round(np.std(similarity_scores), 3))

    for k, v in mean_std_dict.items():
        print(k, v)


def box_plot_similarity(dir_dict, r_level=0, path_name=''):
    similarity_dict = get_similarity_by_dir(dir_dict, r_level)
    similarity_type_data = {'table_to_table': [], 'table_to_image': [],
                            'image_to_image': [], 'total': []}
    similarity_type_labels = {'table_to_table': [], 'table_to_image': [],
                              'image_to_image': [], 'total': []}
    for k, modal_dict in similarity_dict.items():
        for similarity_type, similarity_scores in modal_dict.items():
            similarity_type_data[similarity_type].append(similarity_scores)
            similarity_type_labels[similarity_type].append(k)
    for similarity_type in similarity_type_data:
        plt.figure(figsize=(10, 6))
        plt.boxplot(similarity_type_data[similarity_type], labels=similarity_type_labels[similarity_type], vert=False)
        name = similarity_type
        plt.title(name)
        plt.xlabel('Similarity Score')
        plt.ylabel('Group Type')
        plt.tight_layout()

        path = f'./plots/similarity_score_box_plots/{path_name}/'
        os.makedirs(path, exist_ok=True)
        plt.savefig(f'{path}/{name}.png')
        # plt.show()
        plt.close('all')


def box_plot_generated_questions_similarity():
    # generations = {'zero': 3, 'zero_with_desc': 1, 'multi': 3, 'multi_with_desc': 3}
    generations = {'multi_with_desc': 3}
    id_to_dir = get_id_to_dir(DIR_WIKI_TABLES)
    similarity_data = []
    for generation, q_num in generations.items():
        print("---------------------------------")
        print(f"Generations: {generation}")
        # questions_file = f"./generations/{generation}_turbo.json"
        questions_file = f"../generation/{generation}_turbo_filtered.json"
        with open(questions_file, "r") as f:
            questions = json.load(f)

        scores = get_similarity_generated_questions(questions, id_to_dir)
        scores_list = list(scores.values())
        similarity_data.append(scores_list)

    plt.figure(figsize=(10, 6))
    plt.boxplot(similarity_data, labels=list(generations.keys()), vert=False)
    plt.title('Generations scores')
    plt.xlabel('Similarity Score')
    plt.ylabel('Generation Type')
    plt.tight_layout()
    os.makedirs('./plots/similarity_score_box_plots', exist_ok=True)
    # plt.savefig(f'./plots/similarity_score_box_plots/generations_score.png')
    plt.savefig(f'./plots/similarity_score_box_plots/generations_filtered_score.png')
    # plt.show()
    plt.close('all')


def plot_threshold_vs_eligibility(dir_dict, r_level=0, threshold_step=0.001, path_name=''):
    similarity_dict = get_similarity_by_dir(dir_dict, r_level)
    similarity_type_data = {'table_to_table': [], 'table_to_image': [],
                            'image_to_image': [], 'total': []}
    similarity_type_labels = {'table_to_table': [], 'table_to_image': [],
                              'image_to_image': [], 'total': []}
    for k, modal_dict in similarity_dict.items():
        for similarity_type, scores_list in modal_dict.items():
            scores_list.sort()
            eligible_num = []
            cnt = 0
            for i in np.arange(1, -threshold_step, -threshold_step):
                while cnt < len(scores_list) and scores_list[len(scores_list) - cnt - 1] >= i:
                    cnt += 1
                eligible_num.append(cnt / len(scores_list))

            similarity_type_data[similarity_type].append(eligible_num)
            similarity_type_labels[similarity_type].append(k)
    for similarity_type in similarity_type_data:
        plt.figure(figsize=(12, 7))
        thresholds = np.arange(1, -threshold_step, -threshold_step)
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '#FF7F0E', '#9467BD', '#8C564B']
        styles = ['-', '--', ':', '-.']
        cnt = 0
        for data, label in zip(similarity_type_data[similarity_type], similarity_type_labels[similarity_type]):
            plt.plot(thresholds, data, label=label, color=colors[cnt % len(colors)], linestyle=styles[cnt // len(colors)])
            cnt += 1
        plt.title(similarity_type)
        plt.xlabel('Similarity Score Threshold')
        plt.ylabel('Eligible Number')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title="Legend")
        plt.tight_layout()
        path = f'./plots/similarity_score_threshold_vs_eligibility/{path_name}'
        os.makedirs(path, exist_ok=True)
        plt.savefig(f'{path}/{similarity_type}.png')
        # plt.show()
        plt.close('all')


def generate_plots():
    subsub_to_dir = get_subsub_to_dir(DIR_WIKI_TABLES)
    subject_to_dir = get_subject_to_dir(DIR_WIKI_TABLES)
    total_to_dir = {'total': DIR_WIKI_TABLES}

    box_plot_similarity(subsub_to_dir, 1, path_name='sub-category')
    box_plot_similarity(subject_to_dir, 2, path_name='category')
    box_plot_similarity(total_to_dir, 3, path_name='all')

    plot_threshold_vs_eligibility(subsub_to_dir, 1, path_name='sub-category')
    plot_threshold_vs_eligibility(subject_to_dir, 2, path_name='category')
    plot_threshold_vs_eligibility(total_to_dir, 3, path_name='all')

    box_plot_generated_questions_similarity()


def get_generated_correlation_info():
    generations = {'zero': 3, 'zero_with_desc': 1, 'multi': 3, 'multi_with_desc': 3}

    for generation, q_num in generations.items():
        print("---------------------------------")
        print(f"Generations: {generation}")
        questions_file = f"./generations/{generation}_turbo.json"
        with open(questions_file, "r") as f:
            questions = json.load(f)

        sheet_folder = f"./sheets/{generation}"
        eval_q = read_and_aggregate_csvs(sheet_folder, '*.xlsx')
        get_similarity_correlation(eval_q, questions, q_num)


def get_mean_std_info():
    subsub_to_dir = get_subsub_to_dir(DIR_WIKI_TABLES)
    subject_to_dir = get_subject_to_dir(DIR_WIKI_TABLES)

    print('subsub: ')
    get_similarity_mean_std(subsub_to_dir, 1)

    print("---------------------------------")
    print('subject: ')
    get_similarity_mean_std(subject_to_dir, 2)


def remove_previous(origin, previous):
    origin = sorted(origin, key=lambda d: (d['i'], d['j']))
    previous = sorted(previous, key=lambda d: (d['i'], d['j']))
    result = []
    pnt = 0
    for row in origin:
        if pnt >= len(previous) or row['i'] != previous[pnt]['i'] or row['j'] != previous[pnt]['j']:
            result.append(row)
        else:
            pnt += 1
    return result

def similarity_filter_group_by_topic(bounds, gen_num_each=110, table_threshold=512, previous_pairs=None):
    subject_to_dir = get_subject_to_dir(DIR_WIKI_TABLES)
    cat_res, table_len_dict = get_similarity_full_by_dir(subject_to_dir, 2)

    # res = get_similarity_full_by_dir({'total': DIR_WIKI_TABLES}, 3)['total']
    # res.pop('total')

    cat_modalities = {cat: {'table_to_table': [], 'table_to_image': [],
                            'image_to_image': []} for cat in cat_res}
    total_mod = {'table_to_table': 0, 'table_to_image': 0,
                            'image_to_image': 0, 'total': 0}
    for cat, res in cat_res.items():
        res.pop('total')
        total = 0
        for similarity_type, scores_list in res.items():
            tmp = filter_scores(scores_list, bounds[similarity_type][0],
                                                     bounds[similarity_type][1], table_len_dict=table_len_dict,
                                                     table_threshold=table_threshold)
            if previous_pairs is not None:
                tmp = remove_previous(tmp, previous_pairs[cat][similarity_type])
            cat_modalities[cat][similarity_type] = tmp
            res[similarity_type] = len(tmp)
            total_mod[similarity_type] += res[similarity_type]
            total += res[similarity_type]
        res['total'] = total
        total_mod['total'] += total

    print(cat_res)
    print('total: ', total_mod)
    final_modalities = []
    for mod_type in ['table_to_table', 'table_to_image', 'image_to_image']:
        q_per_topic = gen_num_each
        for i, (topic, mods) in enumerate(cat_modalities.items()):
            num_to_get = min(q_per_topic, len(mods[mod_type]))
            remain = q_per_topic - num_to_get
            if i < len(cat_modalities) - 1:
                q_per_topic += remain // (len(cat_modalities) - i - 1)
            selected = []
            record_dict = {}
            limit = 2
            for i_iter in range(num_to_get):
                while(True):
                    s_ch = False
                    for ii in range(10):
                        sel_ind = np.random.randint(len(mods[mod_type]))
                        sel = mods[mod_type][sel_ind]
                        if sel['wikidata_id'] in record_dict and record_dict[sel['wikidata_id']] >= limit:
                            continue
                        s_ch = True
                        break
                    if not s_ch:
                        limit += 1
                        continue
                    break
                if sel['wikidata_id'] in record_dict:
                    record_dict[sel['wikidata_id']] += 1
                else:
                    record_dict[sel['wikidata_id']] = 0
                mods[mod_type].remove(sel)
                selected.append(sel)
            mods[mod_type] = selected
            # indices = np.random.choice(len(mods[mod_type]), num_to_get, replace=False)
            # mods[mod_type] = list(np.array(mods[mod_type])[indices])
            if mod_type == 'table_to_table':
                final_modalities += [{"wikidata_id": q['wikidata_id'], "table_ids": [q['i'], q['j']], "charts": []} for q in
                               mods[mod_type]]
            elif mod_type == 'table_to_image':
                final_modalities += [{"wikidata_id": q['wikidata_id'], "table_ids": [q['i']], "charts": [q['j']]} for q in
                               mods[mod_type]]
            elif mod_type == 'image_to_image':
                final_modalities += [{"wikidata_id": q['wikidata_id'], "table_ids": [], "charts": [q['i'], q['j']]} for q in
                               mods[mod_type]]
    if previous_pairs is not None:
        for topic, mods in cat_modalities.items():
            for similarity_type, sim_list in mods.items():
                sim_list += previous_pairs[topic][similarity_type]

    return final_modalities, cat_modalities


def check_with_automated_evaluation_multi(simple_format=False, simple_path=None):

    if not simple_format:
        folder = './sheets/multi_with_desc'
        result_df = read_and_aggregate_csvs(folder, '*.xlsx')

        auto_file = "./generations/multi_with_desc_turbo_with_evaluation.json"
        with open(auto_file, "r") as f:
            auto_evals = json.load(f)

        info_tp = 0
        info_fp = 0
        info_tn = 0
        info_fn = 0
        answer_tp = 0
        answer_fp = 0
        answer_tn = 0
        answer_fn = 0

        correct_wrong_dict = {'correct': 'yes', 'wrong': 'no', 'impossible': 'no'}
        yes_no_dict = {'yes.': 'yes', 'yes': 'yes', 'no.': 'no', 'no': 'no', 'yes,': 'yes', 'no,': 'no'}
        for q in auto_evals:
            if ('answer-correct' in q['InternVL2-evaluator'] and 'Q3_answer_correctness' in result_df[q['wikidata_id']]
                    and result_df[q['wikidata_id']]['Q3_answer_correctness'] in ['correct', 'wrong', 'impossible']):
                eval_response = yes_no_dict[q['InternVL2-evaluator']['answer-correct'].lower()[:3]]
                annot_response = correct_wrong_dict[result_df[q['wikidata_id']]['Q3_answer_correctness']]
                if eval_response == 'yes' and annot_response == 'yes':
                    answer_tp += 1
                elif eval_response == 'yes' and annot_response == 'no':
                    answer_fp += 1
                elif eval_response == 'no' and annot_response == 'yes':
                    answer_fn += 1
                elif eval_response == 'no' and annot_response == 'no':
                    answer_tn += 1
            if ('enough-information' in q['InternVL2-evaluator'] and 'Q3_enough_information_to_answer' in result_df[q['wikidata_id']]
                    and result_df[q['wikidata_id']]['Q3_enough_information_to_answer'] in ['no', 'yes']):
                eval_response = yes_no_dict[q['InternVL2-evaluator']['enough-information'].lower()[:3]]
                annot_response = result_df[q['wikidata_id']]['Q3_enough_information_to_answer']
                if eval_response == 'yes' and annot_response == 'yes':
                    info_tp += 1
                elif eval_response == 'yes' and annot_response == 'no':
                    info_fp += 1
                elif eval_response == 'no' and annot_response == 'yes':
                    info_fn += 1
                elif eval_response == 'no' and annot_response == 'no':
                    info_tn += 1
    else:
        with open(simple_path, "r") as f:
            qs = json.load(f)
        info_tp = 0
        info_fp = 0
        info_tn = 0
        info_fn = 0
        answer_tp = 0
        answer_fp = 0
        answer_tn = 0
        answer_fn = 0
        correct_wrong_dict = {'correct': True, 'wrong': False, 'edit': False}
        yes_no_dict = {'yes.': True, 'yes': True, 'no.': False, 'no': False, 'yes,': True, 'no,': False}
        for q in qs:
            try:
                info_eval_response = yes_no_dict[q['InternVL2-evaluator']['enough-information'].lower()[:3]]
                info_annot_response = q['annotation']['valid']
                corr_eval_response = False if "answer-correct" not in q['InternVL2-evaluator'] else q['InternVL2-evaluator']["answer-correct"]
                corr_annot_response = False if not q['annotation']['valid'] else correct_wrong_dict[q['annotation']['vote']]
            except Exception as e:
                print(e)
                continue
            if corr_eval_response and corr_annot_response:
                answer_tp += 1
            elif corr_eval_response and not corr_annot_response:
                answer_fp += 1
            elif not corr_eval_response and corr_annot_response:
                answer_fn += 1
            elif not corr_eval_response and not corr_annot_response:
                answer_tn += 1

            if info_eval_response and info_annot_response:
                info_tp += 1
            elif info_eval_response and not info_annot_response:
                info_fp += 1
            elif not info_eval_response and info_annot_response:
                info_fn += 1
            elif not info_eval_response and not info_annot_response:
                info_tn += 1

    print('Information check:')
    print('Total number:', info_tp + info_tn + info_fn + info_fp)
    print(f'Accuracy (%): {round((info_tp + info_tn) / (info_tp + info_fp + info_tn + info_fn) * 100, 2)}')
    print(f'Precision (%): {round(info_tp / (info_tp + info_fp) * 100, 2)}')
    print(f'Recall (%): {round(info_tp / (info_tp + info_fn) * 100, 2)}')
    print(f'F1 (%): {round(2 * info_tp / (2 * info_tp + info_fp + info_fn) * 100, 2)}')
    print('------------------------------------')
    print('Answer check')
    print('Total number:', answer_tp + answer_tn + answer_fn + answer_fp)
    print(f'Accuracy (%): {round((answer_tp + answer_tn) / (answer_tp + answer_fp + answer_tn + answer_fn) * 100, 2)}')
    print(f'Precision (%): {round(answer_tp / (answer_tp + answer_fp) * 100, 2)}')
    print(f'Recall (%): {round(answer_tp / (answer_tp + answer_fn) * 100, 2)}')
    print(f'F1 (%): {round(2 * answer_tp / (2 * answer_tp + answer_fp + answer_fn) * 100, 2)}')


def filter_scores(score_list, start, end, table_len_dict=None, table_threshold=None):
    new_list = []
    for row in score_list:
        if start <= row['score'] <= end:
            if table_threshold is None or table_threshold is None:
                new_list.append(row)
            wikidata_id = row['wikidata_id']
            if row['i'] in table_len_dict[wikidata_id] and table_len_dict[wikidata_id][row['i']] < table_threshold:
                continue
            if row['j'] in table_len_dict[wikidata_id] and table_len_dict[wikidata_id][row['j']] < table_threshold:
                continue
            new_list.append(row)

    return new_list


def get_filtered_modalities(bounds, num_indices_per_modandcat=6):
    subject_to_dir = get_subject_to_dir(DIR_WIKI_TABLES)
    cat_res = get_similarity_full_by_dir(subject_to_dir, 2)

    # res = get_similarity_full_by_dir({'total': DIR_WIKI_TABLES}, 3)['total']
    # res.pop('total')

    modalities = []
    for cat, res in cat_res.items():
        res.pop('total')
        for similarity_type, scores_list in res.items():
            res[similarity_type] = filter_scores(scores_list, bounds[similarity_type][0],
                                                 bounds[similarity_type][1])
            indices = np.random.choice(len(res[similarity_type]), num_indices_per_modandcat, replace=False)
            res[similarity_type] = list(np.array(res[similarity_type])[indices])
            if similarity_type == 'table_to_table':
                modalities += [{"wikidata_id": q['wikidata_id'], "table_ids": [q['i'], q['j']], "charts": []} for q in res[similarity_type]]
            elif similarity_type == 'table_to_image':
                modalities += [{"wikidata_id": q['wikidata_id'], "table_ids": [q['i']], "charts": [q['j']]} for q in res[similarity_type]]
            elif similarity_type == 'image_to_image':
                modalities += [{"wikidata_id": q['wikidata_id'], "table_ids": [], "charts": [q['i'], q['j']]} for q in res[similarity_type]]
    return modalities

def new_json_acc(path):
    with open(path, "r") as f:
        q_jsons = json.load(f)

    res_dict = {'invalid': 0, 'correct': 0, 'wrong': 0, 'edit': 0}
    for q in q_jsons:
        annotation = q['annotation']
        if annotation['valid']:
            res_dict[annotation['vote']] += 1
        else:
            res_dict['invalid'] += 1
    print(res_dict)
    sum = 0
    for cat, cnt in res_dict.items():
        sum += cnt
    for cat, cnt in res_dict.items():
        print(cat, cnt / sum)


def read_qs(path):
    with open(path, "r") as f:
        previous_generated_questions = json.load(f)
        q_list = [question for question in previous_generated_questions]
    return q_list

def filter_info_with_evaluates(path):
    qs = read_qs(path)
    yes_no_dict = {'yes.': True, 'yes': True, 'no.': False, 'no': False, 'yes,': True, 'no,': False}
    corr = 0
    total = 0
    exceptions = 0
    for q in qs:
        try:
            response = yes_no_dict[q["InternVL2-evaluator"]["enough-information"].lower()[:3]]
            corr += response
            total += 1
        except Exception as e:
            exceptions += 1
            print(e)
    print('correct:', corr, corr / total)
    print('wrong:', total - corr, (total - corr) / total)
    print(exceptions)
