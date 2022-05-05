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
    random_seed = 100

    file_path = "C:\\Users\\Lenovo\\Desktop\\High Dimention Data"

    base, d = vecs_io.fvecs_read(file_path + '\\siftsmall\\siftsmall_base.fvecs')
    index = faiss.index_factory(int(base.shape[1]), "NSG32_PQ16")
    hnsw = index.hnsw
    xq, d = vecs_io.fvecs_read(file_path + "\\siftsmall\\siftsmall_query.fvecs")
    nq, d = xq.shape
    gt, d = vecs_io.ivecs_read(file_path + "\\siftsmall\\siftsmall_groundtruth.ivecs")
    nsg = index.nsg
    time_begin = time.time()
    index.train(base)
    index.add(base)
    time_end = time.time()
    print(time_end - time_begin)
    print("index finsihed ")
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
