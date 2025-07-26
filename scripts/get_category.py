import pywikibot
import json

site = pywikibot.Site("wikidata", "wikidata")
repo = site.data_repository()
pid = "P31"  # instance of
suffix = "_3-10"
with open(f"data/infographics{suffix}.json", "r") as f1, open(
    f"data/category{suffix}.json", "w"
) as f2:
    for line in f1:
        sample = json.loads(line)
        item = pywikibot.ItemPage(repo, sample["qid"])
        try:
            item_dict = item.get()  # Get the data of the item as dict
        except Exception as e:
            print(e)
            continue
        clm_dict = item_dict["claims"]  # Get the claim dictionary
        if pid in clm_dict:
            clm_list = clm_dict[pid]
            properties = []
            for clm in clm_list:
                labels = clm.getTarget().get()["labels"]
                if "en" in labels:
                    properties.append(clm.getTarget().get()["labels"]["en"])
            if len(properties) > 0:
                f2.write(
                    json.dumps(
                        {
                            "qid": sample["qid"],
                            "category": properties,
                        }
                    )
                    + "\n"
                )
