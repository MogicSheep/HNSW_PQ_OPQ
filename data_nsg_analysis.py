import json
import numpy as np
import faiss
import matplotlib.pyplot as plot


def load_json(file_name):
    data = {}
    method = ""
    with open(file_name) as f:
        for line in f:
            line  = line.replace("\n","")
            line_item = line.split(",")
            if line_item[0] == 'nsg128':
                method = "nsg_pq128"
                data[method] = {}
                data[method]["search_time"] = []
            elif line_item[0] == 'nsg64':
                method = "nsg_pq64"
                data[method] = {}
                data[method]["search_time"] = []
            elif line_item[0] == 'nsg32':
                method = "nsg_pq32"
                data[method] = {}
                data[method]["search_time"] = []
            else:
                if len(line_item) < 3:
                    continue
                data[method]["search_time"].append(line_item)

    return data


if __name__ == "__main__":

    for data_file in "result.csv", "none":

        data = load_json(data_file)

        for method, method_data in data.items():
            print(method)
            sub_count = 0
            for k in 1, 10, 50, 100:
                sub_count += 1
                k_qms = []
                k_recall = []
                for item_k in method_data["search_time"]:
                    if item_k[0] == str(k):
                        k_qms.append(item_k[2])
                        k_recall.append(item_k[1])
                m = len(k_qms)

                method_graph_x = np.zeros(m)
                method_graph_y = np.zeros(m)

                for i in range(m):
                    method_graph_x[i] = k_qms[i]
                    method_graph_y[i] = k_recall[i]

                plot.subplot(1, 4, sub_count)
                if method == 'nsg_pq64':
                    plot.plot(method_graph_x, method_graph_y, '-', marker='o', color='green', label='nsg_pq64')
                elif method == 'nsg_pq128':
                    plot.plot(method_graph_x, method_graph_y, '-', marker='^', color='blue', label='nsg_pq128')
                elif method == 'nsg_pq32':
                    plot.plot(method_graph_x, method_graph_y, '-', marker='d', color='pink', label='nsg_pq32')
                elif method == 'hnsw_opq':
                    plot.plot(method_graph_x, method_graph_y, '-', marker='x', color='yellow', label='hnsw_opq')
                elif method == 'hnsw_opq64':
                    plot.plot(method_graph_x, method_graph_y, '-', marker='v', color='red', label='hnsw_opq64')
                plot.xlabel('Qms', fontsize=12)
                plot.ylabel('Recall', fontsize=12)
                # 横轴纵轴命名及字体大小
                plot.legend(loc="upper right", fontsize=12)

        ax = plot.gca()
        plot.show()
