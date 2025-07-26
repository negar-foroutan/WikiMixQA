from argparse import ArgumentParser
import os
import re

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--output_file", type=str, required=True)
    argparser.add_argument("--folder_path", type=str, required=True)
    args = argparser.parse_args()

with open(args.output_file, 'w') as f:
    for topic in os.listdir(args.folder_path):
        if os.path.isdir(os.path.join(args.folder_path, topic)):
            for subtopic in os.listdir(f'{args.folder_path}/{topic}'):
                if os.path.isdir(os.path.join(args.folder_path, topic, subtopic)):
                    for qid in os.listdir(f'{args.folder_path}/{topic}/{subtopic}'):
                        if re.match(r'^Q\d+$', qid):
                            f.write(f"{qid}\t{args.folder_path}/{topic}/{subtopic}\n")
                        else:
                            print(f"not a Wikipedia articles: {qid}")
