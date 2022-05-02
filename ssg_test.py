import numpy as np
import pyssg

data = np.fromfile("./data/sift1M/sift_base.fvecs", dtype=np.float32)
dim = data[0].view(np.int32)
data = data.reshape(-1, dim + 1)
data = np.ascontiguousarray(data[:, 1:])
ndata, dim = data.shape

pyssg.set_seed(1234)
index = pyssg.IndexSSG(dim, ndata)
index.load("/path", data)

k, l = 100, 300
query = np.random.randn(dim).astype(np.float32)
knn = index.search(query, k, l)
print(knn)