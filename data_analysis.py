import json
import numpy as np
import faiss
import matplotlib.pyplot  as plot


def load_json(file_name):
    with open(file_name, "r", encoding="utf-8") as fp:
        return json.load(fp)


if __name__ == "__main__":

    data = load_json("data_log_glove.json")

    for method, method_data in data.items():
        print(method)
        sub_count = 0
        for k in 5, 10, 50, 100:
            sub_count += 1
            k_qms = []
            k_recall = []
            for search_item in method_data["search_time"]:
                for item_k in  search_item:
                    if item_k[0] == k:
                        k_qms.append(item_k[1])
                        k_recall.append(item_k[2])
            m = len(k_qms)

            method_graph_x = np.zeros(m)
            method_graph_y = np.zeros(m)

            for i in range(m):
                method_graph_x[i] = k_qms[i]
                method_graph_y[i] = k_recall[i]

            plot.subplot(1, 4, sub_count)
            if method == 'hnsw':
                plot.plot(method_graph_y, method_graph_x, '-', marker='o', color='green', label='hnsw')
            elif method == 'hnsw_sq':
                plot.plot(method_graph_y, method_graph_x, '-', marker='^', color='blue', label='hnsw_sq')
            else:
                plot.plot(method_graph_y, method_graph_x, '-', marker='d', color='pink', label='hnsw_pq')
            plot.xlabel('Qms', fontsize=12)
            plot.ylabel('Recall', fontsize=12)
            # 横轴纵轴命名及字体大小
            plot.legend(loc="lower left", fontsize=12)

    ax = plot.gca()
    plot.show()
