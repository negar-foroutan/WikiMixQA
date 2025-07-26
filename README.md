<p align="center">
  <!-- <a href="">
    <img alt="Data" src="https://img.shields.io/badge/🤗%20Hugging%20Face-Model-purple">
  </a> -->
  <a href="https://arxiv.org/abs/2506.15594">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-Paper-red">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img alt="License" src="https://img.shields.io/badge/License-MIT-green.svg">
  </a>
</p>

# WikiMixQA: A Multimodal Benchmark for Question Answering over Tables and Charts

<p align="center">
  <img src="assets/wikimix_process.png" alt="Model Overview" width="1000"/>
</p>

<!-- [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)  -->

Code an model for paper: "[WikiMixQA: A Multimodal Benchmark for Question Answering over Tables and Charts](https://arxiv.org/abs/2506.15594)" [ACL 2025 - Findings]


# Data
Please contact the authors to get access to the data (we'll release it soon!)
# Evaluation

### Blind evaluation (i.e., without any context provided)
```
python scripts/evaluate_gpt.py \
    --model-name gpt-4o	 \
    --qa-type blind \
    --input-file Annotations/final_questions.json \
    --output-dir results
```

### Wikidoc evaluation (i.e., with screenshots of the full Wikipedia page)
```
python scripts/evaluate_gemini.py \
    --model-name gpt-4o	 \
    --qa-type wikidoc \
    --input-file Annotations/final_questions.json \
    --output-dir results
```
### Oracle evaluation (i.e., with the two modalities)
```
python scripts/evaluate_gemini.py \
    --model-name gpt-4o	 \
    --qa-type oracle \
    --input-file Annotations/final_questions.json \
    --output-dir results
```


## Evaluation with InternVL2

### Setup

Install dependencies
```
pip install vllm
```

### Oracle evaluation (i.e., with the two modalities)
```
python scripts/evaluate_vllm.py \
    --model-name "OpenGVLab/InternVL2-2B"	 \
    --qa-type oracle \
    --input-file Annotations/final_questions_1001.json \
    --output-dir results_1001 \
    --qid-to-path ../final_new_2/qid_to_path.tsv \
    --num-gpus 8
```


### Wikidoc evaluation (i.e., with screenshots of the full Wikipedia page)

Note that we limit the number of screenshot to the first 15 images for VRAM constraints. 
We can only compute it for 2B model.
```
python scripts/evaluate_vllm.py \
    --model-name "OpenGVLab/InternVL2-2B"	 \
    --qa-type wikidoc \
    --input-file Annotations/final_questions.json \
    --output-dir results \
    --qid-to-path ../final_new_2/qid_to_path.tsv
```

### Blind evaluation (i.e., without any context provided)
```
python scripts/evaluate_gemini.py \
    --model-name "OpenGVLab/InternVL2-2B"	 \
    --qa-type blind \
    --input-file Annotations/final_questions.json \
    --output-dir results \
    --qid-to-path ../final_new_2/qid_to_path.tsv
```

## Evaluation with QwenVL-2


### Oracle evaluation (i.e., with the two modalities)
```
python scripts/evaluate_vllm.py \
    --model-name "Qwen/Qwen2-VL-7B-Instruct"	 \
    --qa-type oracle \
    --input-file Annotations/final_questions.json \
    --output-dir results \
    --qid-to-path ../final_new_2/qid_to_path.tsv \
    --num-gpus 4
```

```
python scripts/evaluate_vllm.py \
    --model-name "Qwen/Qwen2-VL-72B-Instruct"	 \
    --qa-type oracle \
    --input-file Annotations/final_questions.json \
    --output-dir results \
    --qid-to-path ../final_new_2/qid_to_path.tsv \
    --num-gpus 8
```



## Evaluation with Llama-3.2-11B-Vision-Instruct

```
python scripts/evaluate_vllm.py \
    --model-name "meta-llama/Llama-3.2-11B-Vision-Instruct"	 \
    --qa-type oracle \
    --input-file Annotations/final_questions.json \
    --output-dir results \
    --qid-to-path ../final_new_2/qid_to_path.tsv \
    --num-gpus 8
```


# WTabHTML: HTML Wikitables extractor 

### Input:
- Wikipedia HTML dump
- Language

### Output:
File format: JSON list. Each line is a json object of
```
{
    title: wikipedia title
    wikidata: wikidata ID
    url: the url that link to Wikipedia page
    index: the index of table in the Wikipedia page
    html: html content of table
    caption: table caption
    aspects: (Hierachy sections of Wikipedia)  
}
```

### Usage:
#### Download, Extract, and dump wikitables in CR language
```shell
python wtabhtml.py dump -l cr
```

#### Download, Extract, dump wikitables, and generate table images in CR language 

```shell
python wtabhtml.py gen-images -l cr -n 3
```
Note: User can download our [preprocessed dumps](https://drive.google.com/drive/folders/1wU5zdHcb3egxpwyluZCqVBIZnSanUwqN?usp=sharing) then, copy all {LANGUAGE}.jsonl.bz2 (the wikitables dump in PubTabNet format) to `wtabhtml/data/models/wikitables_html_pubtabnet` to generate photo images faster.


If user want to re-run all pipeline, the tool will download Wikipedia HTML dump, extract wikitables, and dump it to `wtabhtml/data/models/wikitables_html_pubtabnet\{LANGUAGE}.jsonl.bz2` file as the following pipeline.

#### Pipeline of Wikitable processing in cr language
```shell
# Download dump
python wtabhtml.py download -l cr
# Parse dump and save json file
python wtabhtml.py parse -l cr
# Read dump
python wtabhtml.py read -l 1 -i ./data/models/cr.jsonl.bz2
# Generate images
python wtabhtml.py gen-images -l cr -n 3
```


### Download images

```bash
python scripts/download_images.py
```

Convert SVG to PNG
```bash
python scripts/convert_svg.py
```

### Getting Wikipedia categories from P31 
```bash
python scripts/get_category.py
```

### Merge files for each topic/subtopics
```bash
python scripts/merge_files.py
```

### Getting HTML pages and generating images from HTML pages

Get raw HTML pages for each article:
```bash
topic="Wikimedia"
subtopic="Person"
mkdir -p html/$topic/$subtopic
python scripts/get_wiki_html.py --input-file "data/${topic}_${subtopic}.json" --output-dir "html/$topic/$subtopic/" 
```

Generate the image from HTML pages:
```bash
python scripts/html2image.py --input-file "data/${topic}_${subtopic}.json" --output-dir "html/$topic/$subtopic/" 
```

Predicting whether the images are likely to be a chart based on the filenames:
```bash
python scripts/predict_chart_or_not.py --input-file "data/${topic}_${subtopic}.json" --output-file "data/${topic}_${subtopic}_chart.json"
```

Compute statistics about the images
```bash
python scripts/category_image_stats.py
```

### Fetching Wiki text

```bash
python scripts/extract_wiki_text.py
```

### Create folder with final images
```bash
python scripts/move_images_to_folder.py
```

### Create folder with tables
```bash
python scripts/move_tables_to_folder.py
```

### Create final dataset
```bash
python scripts/create_final_dataset.py
```

### Split large images generated from HTML pages
```bash
python scripts/split_large_images.py --input-dir final
```

### Extract HTML tables from raw article HTML
```bash
python scripts/extract_tables_from_html.py --input-dir final
```

### Generate images from HTML tables
First, install [geckodriver](https://github.com/mozilla/geckodriver/releases):
```bash
wget https://github.com/mozilla/geckodriver/releases/download/v0.34.0/geckodriver-v0.34.0-linux64.tar.gz
tar -xvzf geckodriver-v0.34.0-linux64.tar.gz
```

Then, install Firefox from APT:
```bash
sudo snap remove firefox
sudo add-apt-repository ppa:mozillateam/ppa
echo '
Package: *
Pin: release o=LP-PPA-mozillateam
Pin-Priority: 1001
' | sudo tee /etc/apt/preferences.d/mozilla-firefox
echo 'Unattended-Upgrade::Allowed-Origins:: "LP-PPA-mozillateam:${distro_codename}";' | sudo tee /etc/apt/apt.conf.d/51unattended-upgrades-firefox
sudo apt install firefox
```

Finally, generate the images
```bash
# add path to geckodriver
export PATH=$PATH:. 
python scripts/gen_images_from_html_tables.py --input-dir final
```

### Add extra information about the images with gpt-4-vision-preview

Getting details for each articles:
```bash
topic="Economy"
subtopic="Stock market"
python scripts/predict_chart_information.py \
    --data-dir "data" \
    --output-dir "final" \
    --topic $topic \
    --subtopic $subtopic 
```

Getting details only for articles with charts:
```bash
python scripts/predict_chart_information.py \
    --data-dir "data" \
    --output-dir "final" \
    --topic $topic \
    --subtopic $subtopic \
    --chart 
```

## Generate table description with LLaMa-3-8b-instruct
```console
python scripts/table-to-text.py \
    --input-dir "final" \
    --checkpoint-dir "/mnt/datastore/models/meta-llama/Meta-Llama-3-8B-Instruct" \
    --dtype fp16 \
    --device "cuda:2"
```

## Compute embeddings for chart and table descriptions
```console
python scripts/compute_embeddings.py \
    --input-dir "final" \
    --model-name "BAAI/bge-reranker-v2-m3" \
    --use-fp16 \
    --device "cuda:2"
```



# Citation

If you use this code for your research, please cite our paper:

``` bib
@inproceedings{foroutan-etal-2025-wikimixqa,
      title={WikiMixQA: A Multimodal Benchmark for Question Answering over Tables and Charts}, 
      author={Negar Foroutan and Angelika Romanou and Matin Ansaripour and Julian Martin Eisenschlos and Karl Aberer and Rémi Lebret},
      year={2025},
      url={https://arxiv.org/abs/2506.15594},
}