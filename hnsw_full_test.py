import json
import time
import sys
import matplotlib.pyplot as plt
import numpy as np
import faiss

try:
    from faiss.contrib.datasets_fb import DatasetSIFT1M, DatasetGlove
except ImportError:
    from faiss.contrib.datasets import DatasetSIFT1M, DatasetGlove


# from datasets import load_sift1M


def save_json(data):
    with open("data_log_sift.json", "w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=4)


def test_sift_1m():
    ds = DatasetSIFT1M()

    xq = ds.get_queries()
    xb = ds.get_database()
    gt = ds.get_groundtruth()
    xt = ds.get_train()

    nq, d = xq.shape

    data_log = {}
    todo = 'hnsw hnsw_sq ivf ivf_hnsw_quantizer nsg'.split()
    # todo = 'nsg_pq'.split()
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

    if 'hnsw' in todo:

        print("Testing HNSW Flat")

        index = faiss.IndexHNSWFlat(d, 32)

        # training is not needed

        # this is the default, higher is more accurate and slower to
        # construct
        index.hnsw.efConstruction = 40

        print("add")
        # to see progress
        index.verbose = True
        data_log['hnsw'] = {}
        t0 = time.time()
        index.add(xb)
        t1 = time.time()
        data_log['hnsw']['index_time'] = (t1 - t0) * 1000.0

        hnsw_x1 = np.zeros(5)
        hnsw_y1 = np.zeros(5)
        print("search")
        count = 0
        data_log['hnsw']['search_time'] = []
        for efSearch in 16, 32, 64, 128, 256:
            for bounded_queue in [True, False]:
                print("efSearch", efSearch, "bounded queue", bounded_queue, end=' ')
                index.hnsw.search_bounded_queue = bounded_queue
                index.hnsw.efSearch = efSearch
                data = evaluate(index)
                res, hnsw_x1[count], hnsw_y1[count] = data[0]
                data_log['hnsw']['search_time'].append(data)
            count += 1

        plt.plot(hnsw_x1, hnsw_y1, '-', marker='o', color='green', label='hnsw_only')

    if 'hnsw_sq' in todo:

        print("Testing HNSW with a scalar quantizer")
        # also set M so that the vectors and links both use 128 bytes per
        # entry (total 256 bytes)
        index = faiss.IndexHNSWSQ(d, faiss.ScalarQuantizer.QT_8bit, 32)

        print("training")
        # training for the scalar quantizer

        # data_log
        data_log['hnsw_pq'] = {}
        data_log['hnsw_pq']['index_time'] = 0
        data_log['hnsw_pq']['search_time'] = []
        t0 = time.time()
        index.train(xb)

        # this is the default, higher is more accurate and slower to
        # construct
        index.hnsw.efConstruction = 40
        hnsw_pq_x1 = np.zeros(5)
        hnsw_pq_y1 = np.zeros(5)
        print("add")
        # to see progress
        index.verbose = True
        index.add(xb)
        t1 = time.time()

        data_log['hnsw_pq']['index_time'] = (t1 - t0) * 1000.0
        print("search")
        count = 0
        for efSearch in 16, 32, 64, 128, 256:
            print("efSearch", efSearch, end=' ')
            index.hnsw.efSearch = efSearch
            data = evaluate(index)
            res, hnsw_pq_x1[count], hnsw_pq_y1[count] = data[0]
            data_log['hnsw_pq']['search_time'].append(data)
            count += 1
        plt.plot(hnsw_pq_x1, hnsw_pq_y1, '-', marker='s', color='blue', label='hnsw_PQ')

    if 'ivf' in todo:

        print("Testing IVF Flat (baseline)")
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, 16384)
        index.cp.min_points_per_centroid = 5  # quiet warning

        # to see progress
        index.verbose = True
        data_log['ivf'] = {}
        data_log['ivf']['index_time'] = 0
        data_log['ivf']['search_time'] = []
        print("training")
        t0 = time.time()
        index.train(xt)

        print("add")
        index.add(xb)
        ivf_x1 = np.zeros(5)
        ivf_y1 = np.zeros(5)
        t1 = time.time()
        data_log['ivf']['index_time'] = (t1 - t0) * 1000.0
        print("search")
        count = 0
        for nprobe in 1, 4, 16, 64, 256:
            print("nprobe", nprobe, end=' ')
            index.nprobe = nprobe
            data = evaluate(index)
            res, ivf_x1[count], ivf_y1[count] = data[0]
            data_log['ivf']['search_time'].append(data)
            count += 1
        plt.plot(ivf_x1, ivf_y1, '-', marker='D', color='yellow', label='ivf')

    if 'nsg' in todo:

        print("Testing NSG Flat")

        index = faiss.IndexNSGFlat(d, 32)
        index.build_type = 1
        # training is not needed

        # this is the default, higher is more accurate and slower to
        # construct
        data_log['nsg'] = {}
        data_log['nsg']['search_time'] = []
        data_log['nsg']['index_time'] = 0
        t0 = time.time()
        print("add")
        # to see progress
        index.verbose = True
        index.add(xb)
        nsg_x1 = np.zeros(5)
        nsg_y1 = np.zeros(5)
        t1 = time.time()
        data_log['nsg']['index_time'] = (t1 - t0) * 1000.0

        print("search")
        count = 0
        for search_L in 16, 32, 64, 128, 256:
            print("search_L", search_L, end=' ')
            index.nsg.search_L = search_L
            data = evaluate(index)
            res, nsg_x1[count], nsg_y1[count] = data[0]
            data_log['nsg']['search_time'].append(data)
            count += 1
        plt.plot(nsg_x1, nsg_y1, '-', marker='^', color='red', label='nsg')


    plt.xlabel('Qms', fontsize=12)
    plt.ylabel('Recall', fontsize=12)
    # 横轴纵轴命名及字体大小
    plt.legend(loc="lower left", fontsize=12)
    # 图例的位置

    ax = plt.gca()

    plt.savefig('SIFT1M_test_k_all' + '.pdf', bbox_inches='tight', dpi=300, pad_inches=0.0)
    plt.show()
    # 将绘制的图另存为目录下的.pdf格式文件;tight和pad_inches的设置是为了在保存图片时去除白边
    save_json(data_log)


if __name__ == "__main__":
    k = int(sys.argv[1])
    print(k)
    todo = sys.argv[2:]

    print("load data")

    test_sift_1m()
