import os


def get_id_to_dir(data_dir):
    id_to_dir = {}
    for subject in os.listdir(data_dir):
        dir_sub = f"{data_dir}/{subject}"
        for subsub in os.listdir(dir_sub):
            dir_subsub = f"{dir_sub}/{subsub}"
            for id_ in os.listdir(dir_subsub):
                id_to_dir[id_] = f"{dir_subsub}/{id_}"
    return id_to_dir


def get_subsub_to_dir(data_dir):
    subsub_to_dir = {}
    for subject in os.listdir(data_dir):
        dir_sub = f"{data_dir}/{subject}"
        for subsub in os.listdir(dir_sub):
            subsub_to_dir[f'{subject}_{subsub}'] = f"{dir_sub}/{subsub}"
    return subsub_to_dir


def get_subject_to_dir(data_dir):
    subject_to_dir = {}
    for subject in os.listdir(data_dir):
        subject_to_dir[subject] = f"{data_dir}/{subject}"
    return subject_to_dir


class CustomError(Exception):
    """Custom exception class for specific error conditions."""
    pass
