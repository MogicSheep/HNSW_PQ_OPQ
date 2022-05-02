import json
import numpy as np
import faiss
import matplotlib as plot


def load_json():
    with open("data_log.json", "r", encoding="utf-8") as fp:
        return json.load(fp)


if __name__ == "__main__":

    data = load_json()

    for method, method_data in data.items():
        print(method)
