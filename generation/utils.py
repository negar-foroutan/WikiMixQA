import os
import json
import base64


def get_wiki_page_stats_map(data_folder):
  """ Get the stats of the wiki pages """
  wiki_page_stats_map = {}
  main_topics = os.listdir(data_folder)
  for main_topic in main_topics:
    if not os.path.isdir(os.path.join(data_folder, main_topic)):
      continue
    print(f"Main topic: {main_topic}")
    sub_topics = os.listdir(os.path.join(data_folder, main_topic))
    for sub_topic in sub_topics:
      print(f"Sub topic: {sub_topic}")
      QIDs = os.listdir(os.path.join(data_folder, main_topic, sub_topic))
      for QID in QIDs:
        wiki_page_stats_map[QID] = {}
        chart_info_file_path = os.path.join(data_folder, main_topic, sub_topic, QID, "gpt-4-turbo.json")
        data_charts = []
        if os.path.exists(chart_info_file_path):
          with open(chart_info_file_path, "r") as f:
            potential_charts = [json.loads(row) for row in f.readlines()]
            for chart in potential_charts:
              if "choices" in chart:
                try:
                  chart_record = json.loads(chart["choices"][0]["message"]["content"])            
                  if not chart_record["chart"] == "no":
                    data_charts.append(chart)
                except:
                  if "\"chart\": \"yes\"" in chart["choices"][0]["message"]["content"]:
                    data_charts.append(chart)
        else:
          data_charts = []
        data_charts = [row["filename"] for row in data_charts]
        wiki_page_stats_map[QID]["data_charts"] = data_charts
        if QID.endswith(".jsonl") or QID.endswith(".json"):
          continue
        wiki_page_stats_map[QID]["topic"] = (main_topic, sub_topic)
        page_folder_path = os.path.join(data_folder, main_topic, sub_topic, QID)
        wiki_page_stats_map[QID]["path"] = page_folder_path
        with open(os.path.join(page_folder_path, "html_tables.json"), "r") as f:
          tables = [json.loads(row) for row in f.readlines()]
          wiki_page_stats_map[QID]["num_tables"] = len(tables)
        with open(os.path.join(page_folder_path, "metadata.json"), "r") as f:
          metadata = json.load(f)
          wiki_page_stats_map[QID]["num_charts"] = len(data_charts)
          wiki_page_stats_map[QID]["url"] = metadata["url"]
        
  return wiki_page_stats_map

def encode_image(image_path):
  """ Encodes the image to base64 """
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


def get_table_content(table_record, with_desc=False):
    """ Get the content of the given table record """
    table_content = ""
    if with_desc and "desc" in table_record:
      table_content += table_record["desc"] + "\n"
    if "caption" in table_record:
      table_content += table_record["caption"] + "\n"
    table_content += table_record["html"]
    return table_content
