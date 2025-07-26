import json
from mediawiki import MediaWiki
from wikidata.client import Client
from tqdm import tqdm

if __name__ == "__main__":
    client = Client()
    wikipedia = MediaWiki()
    filename = "data/stats.txt"
    qids = []
    print("Loading qid from data/stats.txt")
    with open(filename, "r") as f:
        for line in f:
            qid, count = line.strip().split("\t")
            count = int(count)
            # Only fetch images for pages with more a number of tables between 3 and 10
            if count > 10:
                continue
            if count < 3:
                break
            qids.append(qid)

    with open("data/images_3-10.json", "w") as f, open(
        "data/not_found_3-10.txt", "w"
    ) as f2:
        for qid in tqdm(qids, desc="Fetching images"):
            try:
                entity = client.get(qid, load=True)
                title = entity.attributes["sitelinks"]["enwiki"]["title"]
                url = entity.attributes["sitelinks"]["enwiki"]["url"]
                p = wikipedia.page(title=title, auto_suggest=True)
                images = [i for i in p.images if "Flag" not in i]
                f.write(
                    json.dumps(
                        {"qid": qid, "title": title, "url": url, "images": images}
                    )
                    + "\n"
                )
            except Exception as e:
                print(e)
                print("Not found:", qid)
                f2.write(qid + "\n")
