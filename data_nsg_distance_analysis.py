import json
import numpy as np
import faiss
import matplotlib.pyplot as plot


def load_json(file_name):
    data = {}
    method = ""
    if file_name == "result256.csv":
        method = "NSG_PQ_64_256"
        data[method] = {}
        data[method]["Product_Search"] = []
        data[method]["Progress_Search"] = []
        data[method]["Table_Search"] = []
        data[method]["recall"] = []
    else:
        method = "NSG_PQ_64_4096"
        data[method] = {}
        data[method]["Product_Search"] = []
        data[method]["Progress_Search"] = []
        data[method]["Table_Search"] = []
        data[method]["recall"] = []

    with open(file_name) as f:
        for line in f:
            line = line.replace("\n", "")
            line_item = line.split(",")
            data[method]["recall"].append(line_item[0])
            data[method]["Product_Search"].append(line_item[1])
            data[method]["Progress_Search"].append(line_item[2])
            data[method]["Table_Search"].append(line_item[3])

    return data


if __name__ == "__main__":
    sub_count = 0
    for data_file in "result256.csv", "result4096.csv":
        sub_count += 1
        data = load_json(data_file)
        for method, search_data in data.items():
            recall = []
            PQ = []
            PP = []
            PT = []
            for recall_item in data[method]["recall"]:
                recall.append(recall_item)

            m = len(recall)
            method_graph_x = np.zeros(m)

            for i in range(m):
                method_graph_x[i] = recall[i]

            for PQ_item in data[method]["Product_Search"]:
                PQ.append(PQ_item)
            for PP_item in data[method]["Progress_Search"]:
                PP.append(PP_item)
            for PT_item in data[method]["Table_Search"]:
                PT.append(PT_item)
            method_graph_PQ = np.zeros(m)
            method_graph_PP = np.zeros(m)
            method_graph_PT = np.zeros(m)

            for i in range(m):
                method_graph_PQ[i] = PQ[i]
                method_graph_PP[i] = PP[i]
                method_graph_PT[i] = PT[i]

            plot.subplot(1, 2, sub_count)
            plot.plot(method_graph_x, method_graph_PQ, '-', marker='o', color='green', label='PQ')
            plot.plot(method_graph_x, method_graph_PP, '-', marker='^', color='blue', label='PP')
            plot.plot(method_graph_x, method_graph_PT, '-', marker='d', color='pink', label='PT')
            plot.xlabel('Recall@1', fontsize=12)
            plot.ylabel('Time use s', fontsize=12)
            # 横轴纵轴命名及字体大小
            plot.legend(loc="upper right", fontsize=12)

            ax = plot.gca()
            plot.show()
