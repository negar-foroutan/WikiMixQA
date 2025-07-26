from openai import OpenAI
from argparse import ArgumentParser
import pandas as pd
import json

client = OpenAI()


def get_chart_or_not_prediction(image_filename: str) -> str:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are an image filename reader and you should guess whether the filename describes an image representing a chart or not. You should answer by yes or no in the JSON format.",
            },
            {
                "role": "user",
                "content": f'Filename is "{image_filename}".',
            },
        ],
    )
    return response.choices[0].message.content


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--input-file", type=str, required=True)
    argparser.add_argument("--output-file", type=str, required=True)
    args = argparser.parse_args()

    df = pd.read_json(args.input_file, lines=True)
    print(df.head())
    print(df.info())

    answers = []
    for i, row in df.iterrows():
        images = row["images"]
        charts = []
        for filename in images:
            try:
                prediction = json.loads(get_chart_or_not_prediction(filename))
                print(f"Prediction for {filename}: {prediction['answer']}")
                charts.append(
                    {
                        "filename": filename,
                        "chart": prediction["answer"],
                    }
                )
            except Exception as e:
                print(f"Could not get prediction for {filename}: {e}")
                charts.append(
                    {
                        "filename": filename,
                        "chart": "unknown",
                    }
                )
        answers.append(charts)
    df["images"] = answers
    df.to_json(args.output_file, lines=True, orient="records")
