from argparse import ArgumentParser
from torchtune.models.llama3 import llama3_8b, llama3_tokenizer
from torchtune import utils
from torchtune.data import Message, validate_messages
import torch
import json
import os

categories = json.load(open("data/category_taxonomy.json"))

system_prompt = Message(
    role="system",
    content="You are a table-to-text assistant. I'll give you a table in HTML format, please write a textual description of the table content.",
)


def table_to_text(
    table,
    tokenizer,
    model,
    device,
    max_seq_len=4096,
    max_new_tokens=256,
    temperature=0.6,
    top_k=300,
    custom_generate_next_token=None,
):
    messages = [
        system_prompt,
        Message(role="user", content=table),
        Message(role="assistant", content=""),
    ]
    validate_messages(messages)
    tokens, _ = tokenizer.tokenize_messages(messages, max_seq_len=max_seq_len)
    inputs = torch.tensor(tokens, device=device)
    prompt = inputs[:-3]  # remove eos_it

    with torch.no_grad():
        generated_tokens = utils.generate(
            model=model,
            prompt=prompt,
            max_generated_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            stop_tokens=tokenizer.stop_tokens,
            custom_generate_next_token=custom_generate_next_token,
        )
    generated_text = tokenizer.decode(
        generated_tokens[prompt.size(0) :], truncate_at_eos=False
    )
    return generated_text


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--input-dir", type=str, required=True)
    argparser.add_argument("--checkpoint-dir", type=str, required=True)
    argparser.add_argument("--max-seq-length", type=int, default=4096)
    argparser.add_argument("--max-new-tokens", type=int, default=256)
    argparser.add_argument(
        "--checkpoint-files",
        type=str,
        nargs="+",
        default=["consolidated.00.pth"],
    )
    argparser.add_argument(
        "--dtype", type=str, choices=["fp32", "fp16", "bf16"], default="fp16"
    )
    argparser.add_argument("--device", type=str, default="cuda:0")
    args = argparser.parse_args()

    # Define device and dtype
    device = utils.get_device(args.device)
    dtype = utils.get_dtype(args.dtype)
    # load tokenizer
    tokenizer = llama3_tokenizer(os.path.join(args.checkpoint_dir, "tokenizer.model"))
    # load checkpoint
    checkpointer = utils.FullModelMetaCheckpointer(
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_files=args.checkpoint_files,
        model_type="llama3",
        output_dir="./output",
    )
    # load model from checkpoint
    ckpt_dict = checkpointer.load_checkpoint()
    with utils.set_default_dtype(dtype), utils.get_device(device):
        model = llama3_8b()
    model.load_state_dict(ckpt_dict["model"])
    # validate model was loaded in with the expected dtype
    utils.validate_expected_param_dtype(model.named_parameters(), dtype=dtype)
    print(f"Model is initialized with precision {dtype}.")
    # ensure the cache is setup on the right device
    with utils.get_device(device=device):
        model.setup_caches(max_batch_size=1, dtype=dtype)
    # loop over all tables
    for topic, v in categories.items():
        topic_dir = os.path.join(args.input_dir, topic)
        for subtopic in v.keys():
            subtopic_dir = os.path.join(topic_dir, subtopic)
            for qid in os.listdir(subtopic_dir):
                if not os.path.isdir(os.path.join(subtopic_dir, qid)):
                    continue
                tables_filename = os.path.join(subtopic_dir, qid, "html_tables.json")
                new_tables_filename = os.path.join(
                    subtopic_dir, qid, "html_tables_with_desc.json"
                )
                if os.path.exists(new_tables_filename):
                    print(f"Skipping {qid}")
                    continue
                if not os.path.exists(tables_filename):
                    print(f"Does not exist {qid}")
                    continue
                # read tables from html file
                print(f"Processing {topic}/{subtopic}/{qid}")
                outputs = []
                with open(tables_filename) as f:
                    for line in f:
                        table = json.loads(line)
                        table["desc"] = table_to_text(
                            table["html"],
                            tokenizer,
                            model,
                            device,
                            max_seq_len=args.max_seq_length,
                            max_new_tokens=args.max_new_tokens,
                            temperature=0.6,
                            top_k=300,
                        )
                        outputs.append(table)
                with open(new_tables_filename, "w") as f:
                    for t in outputs:
                        json.dump(t, f)
                        f.write("\n")
