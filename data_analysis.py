import json
import numpy as np
import faiss
import matplotlib.pyplot  as plot


def load_json(file_name):
    with open(file_name, "r", encoding="utf-8") as fp:
        return json.load(fp)

def save_json(data, file_name):
    with open(file_name, "w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=4)


if __name__ == "__main__":

    for data_file in "data_log_sift.json","data_log_glove.json":

        data = load_json(data_file)

        for method, method_data in data.items():
            print(method)
            sub_count = 0
            for k in 1, 5, 10, 50, 100:
                sub_count += 1
                k_qms = []
                k_recall = []
                sorted(method_data["search_time"], key=lambda x: (x[0] == k) * x[2], reverse= True)
                for search_item in method_data["search_time"]:
                    for item_k in search_item:
                        if item_k[0] == k:
                            k_qms.append(item_k[1])
                            k_recall.append(item_k[2])
                m = len(k_qms)

                method_graph_x = np.zeros(m)
                method_graph_y = np.zeros(m)

                for i in range(m):
                    method_graph_x[i] = k_qms[i]
                    method_graph_y[i] = k_recall[i]

                plot.subplot(1, 5, sub_count)
                if method == 'hnsw':
                    plot.plot(method_graph_x, method_graph_y, '-', marker='o', color='green', label='hnsw')
                elif method == 'hnsw_sq':
                    plot.plot(method_graph_x, method_graph_y, '-', marker='^', color='blue', label='hnsw_sq')
                elif method == 'hnsw_pq':
                    plot.plot(method_graph_x, method_graph_y, '-', marker='d', color='pink', label='hnsw_pq')
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
