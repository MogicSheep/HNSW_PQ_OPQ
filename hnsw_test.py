import numpy as np
import faiss
import vecs_io
import time

def get_neighbors(hnsw, i, level):
    " list the neighbors for node i at level "
    assert i < hnsw.levels.size()
    assert level < hnsw.levels.at(i)
    be = np.empty(2, 'uint64')
    hnsw.neighbor_range(i, level, faiss.swig_ptr(be), faiss.swig_ptr(be[1:]))
    return [hnsw.neighbors.at(j) for j in range(be[0], be[1])]


def get_graph(hnsw, level):
    graph = []
    for i in range(hnsw.levels.size()):
        if level >= hnsw.levels.at(i):
            tmp_neighbors = []
        else:
            tmp_neighbors = get_neighbors(hnsw, i, level)
            while -1 in tmp_neighbors:
                tmp_neighbors.remove(-1)
        graph.append(tmp_neighbors)

    return graph


if __name__ == "__main__":
    M = 16
    efConstruction = 100
    efSearch = 100
    random_seed = 100

    print("faiss hnsw, M %d, efConstruction %d, efSearch %d" % (M, efConstruction, efSearch))

    np.random.seed(random_seed)

    file_path = "C:\\Users\\Lenovo\\Desktop\\High Dimention Data\\HNSWpython\\data"


    base, d = vecs_io.fvecs_read(file_path + '\\sift1M\\sift_base.fvecs')
    index = faiss.index_factory(int(base.shape[1]), "HNSW32_PQ16")
    hnsw = index.hnsw
    xq, d = vecs_io.fvecs_read(file_path + "\\sift1M\\sift_query.fvecs")
    nq, d = xq.shape
    gt, d = vecs_io.ivecs_read(file_path + "\\sift1M\\sift_groundtruth.ivecs")
    hnsw.efConstruction = efConstruction
    hnsw.efSearch = efSearch
    time_begin = time.time()
    index.train(base)
    index.add(base)
    time_end = time.time()
    print(time_end - time_begin)


    def evaluate(index):
        # for timing with a single core
        # faiss.omp_set_num_threads(1)
        ans = []
        for k in [1, 5, 10]:
            t0 = time.time()
            D, I = index.search(xq, k)
            t1 = time.time()

            missing_rate = (I == -1).sum() / float(k * nq)
            recall_at_1 = (I == gt[:, :1]).sum() / float(nq)
            print("\t %7.3f ms per query, R@1 %.4f, missing rate %.4f" % (
                (t1 - t0) * 1000.0 / nq, recall_at_1, missing_rate))
            ans.append((k, nq / ((t1 - t0) * 1000.0), recall_at_1))

        return ans

    evaluate(index)
    for i in range(index.hnsw.max_level):
        graph = get_graph(index.hnsw, i)
        max_m = -1
        max_m_idx = -1
        total_edge = 0
        for j, edge in enumerate(graph, 0):
            total_edge += len(edge)
            if len(edge) > max_m:
                max_m = len(edge)
                max_m_idx = j
        print("at level %d, n_edges %d largest M %d idx %d" % (i, total_edge, max_m, max_m_idx))
        if i == 0:
            for edge in graph:
                if len(edge) <= 0:
                    print("the edge should be greater than 0 at bottom level")
