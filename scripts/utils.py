import json
import base64
from google.cloud import storage

def encode_image(image_path):
  """ Encodes the image to base64 """
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
  
def define_prompt(qa_type: str, n_charts: int = 0, n_tables: int = 0):
    if qa_type == "wikidoc":
        return """
            Given the following document, which includes tables and charts, which of the following answer is correct? A, B, C or D. 
            Extract and analyze the relevant data from the document to select the correct answer. If the document does not contain enough information, infer the most plausible answer or state ‘Unable to determine’.
            Please answer in the format A, B, C, D, or ‘Unable to determine’.”
            """
    elif qa_type == "oracle":
        generic_prompt = ", which of the following answer is correct? A, B, C or D. Please answer in the format A, B, C or D."
        if n_charts == 1 and n_tables == 1:
            return f"Given the following chart and table (in HTML format){generic_prompt}"
        elif n_charts == 2:
            return f"Given the following two charts{generic_prompt}"
        elif n_tables == 2:
            return f"Given the following two tables (in HTML format){generic_prompt}"
        else:
            raise ValueError("Invalid number of charts and tables")
    elif qa_type == "blind":
        return """
            Which of the following answer is correct? A, B, C or D. Please answer in the format A, B, C or D.
            If the chart or table is unavailable, please infer the most plausible answer based on general reasoning or state ‘Unable to determine’ if no inference is possible. 
            Please answer in the format A, B, C, D, or ‘Unable to determine’.
            """

def load_html_tables(topic, subtopic, qid):
    # Initialize the GCS client
    client = storage.Client()

    # Get the bucket and blob (file)
    bucket = client.get_bucket("wiki-chart-tables")
    blob = bucket.blob(f"{topic}/{subtopic}/{qid}/html_tables.json")

    # Download the file content as a string
    jsonl_data = blob.download_as_text()

    # Split by lines and parse each JSON object
    records = [json.loads(line) for line in jsonl_data.strip().splitlines()]

    return [record["html"] for record in records]


def load_doc_images(topic, subtopic, qid, delimiter=None):

    storage_client = storage.Client()
    # Note: Client.list_blobs requires at least package version 1.17.0.
    blobs = storage_client.list_blobs("wiki-chart-tables", prefix=f"{topic}/{subtopic}/{qid}/images", delimiter=delimiter)
    # Note: The call returns a response only when the iterator is consumed.
    return [blob.name for blob in blobs]
