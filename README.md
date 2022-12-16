# Single GPU Test

# Single GPU kNN algorithm comparison

- In this test, two primary libraries will be used to test the performance of a single GPU on searching the k-nearest neighbors of a dataset
- The first one is RAPIDS/cuML, that is implemented only for GPU usage(link do rapids)
- The second one is Faiss, that has support for either GPU or CPU.

## How to install

- última coisa a ser feita
- Explicar como baixar o RAPIDS pelo docker
- Explicar como baixar o faiss dentro de um docker ou fora mesmo.
- Explicar requisitos que precisam ter

## Algorithms

### Brute-Force

- This algorithm uses the brute force, in which the real k nearest neighbors are searched, by comparing all the samples
- As it compares all the samples, this algorithm complexity is O(n²)
- Therefore, it is better to use Approximate Nearest Neighbor methods
- This algorithm is Deterministic
    
    Link : [https://www.analyticsvidhya.com/blog/2018/03/introduction-k-neighbours-algorithm-clustering/?ref=morioh.com&utm_source=morioh.com](https://www.analyticsvidhya.com/blog/2018/03/introduction-k-neighbours-algorithm-clustering/?ref=morioh.com&utm_source=morioh.com)
    
    
    
    ![Screenshot from 2022-12-16 10-44-49](https://user-images.githubusercontent.com/91696970/208111841-6e11642f-48c7-4ae8-bd9e-be62674fd127.png)

- This algorithm is implemented
in cuML
- ******************************How to use in Python******************************
    
    ### In cuML
    
    ```python
    import cudf
    from cuml.neighbors import NearestNeighbors
    from cuml.datasets import make_blobs
    
    X, _ = make_blobs(n_samples=5, centers=5,
                      n_features=10, random_state=42)
    
    # build a cudf Dataframe
    X_cudf = cudf.DataFrame(X)
    
    # fit model
    model = NearestNeighbors(n_neighbors=3,algorithm='brute'output_type='numpy')
    model.fit(X)
    
    # get 3 nearest neighbors
    distances, indices = model.kneighbors(X_cudf)
    ```
    
    ### ****************In FAISS****************
    
    ```python
    import numpy as np
    import pdb
    
    import faiss
    
    print("load data")
    
    xb, xq, xt, gt = load_sift1M()
    nq, d = xb.shape
    nlist = 1000
    
    # we need only a StandardGpuResources per GPU
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = 0
    
    index = faiss.GpuIndexFlatL2(res, d, flat_config)
    
    print("add vectors to index")
    
    index.add(xb)
    distances, indices = index.search(xb, 5)
    ```
    

### Inverted file with exact post-verification (IVFF)

- This algorithm divides the dataset in partitions, and performs the search in the relevant partitions (Voronoi cells), with this approach, the brute force method is avoided
- However, as the search is performed only in few partitions, the algorithm estimates the kNN, so it is an Approximate Nearest Neighbor method
- To obtain the partitions, the algorithm use k-Means
- In this algorithm, the tradeoff is between the number of partitions to be searched and time/accuracy
- This method is Probabilistic
    
    Link: [https://www.sciencedirect.com/science/article/abs/pii/S0020025522006806](https://www.sciencedirect.com/science/article/abs/pii/S0020025522006806)
    
   ![Screenshot from 2022-12-16 10-45-41](https://user-images.githubusercontent.com/91696970/208112011-4cc47d61-7d76-4678-99d3-602d80369362.png)
    

- This algorithm is present in the cuML documentation, but cuML is linked with a unstable version of faiss, resulting in lots of erros
- So, in this test the Faiss version will be used
- ****************************************How to use in Python****************************************
    
    ```python
    import numpy as np
    import pdb
    
    import faiss
    
    print("load data")
    
    xb, xq, xt, gt = load_sift1M()
    nq, d = xb.shape
    nlist = 1000
    
    # we need only a StandardGpuResources per GPU
    res = faiss.StandardGpuResources()
    config = faiss.GpuIndexIVFFlatConfig()
    
    #The metric can be changed
    index = faiss.GpuIndexIVFFlat(res, d, nlist, faiss.METRIC_L2,config)
    
    index.train(xb)
    index.add(xb)
    distances, indices = index.search(xb, 5)
    ```
    

### IVF and scalar quantizer (IVFSQ)

- This algorithm is very similar with the IVFPQ algorithm, but one of the differences is that the vectors are transformed in binary representation, to increase the efficiency of the search
- This is an Approximate Nearest Neighbor method
- This is method is Probabilistic
    
    Link: [https://www.sciencedirect.com/science/article/abs/pii/S0020025522006806](https://www.sciencedirect.com/science/article/abs/pii/S0020025522006806)
    
- This algorithm is present in the cuML documentation, but cuML is linked with a unstable version of faiss, resulting in lots of erros
- So, in this test the Faiss version will be used
- ****************************************How to use in Python****************************************
    
    ```python
    import numpy as np
    import pdb
    
    import faiss
    import faiss.contrib.torch_utils
    
    print("load data")
    
    xb, xq, xt, gt = load_sift1M()
    nq, d = xb.shape
    nlist = 1000
    
    # we need only a StandardGpuResources per GPU
    res = faiss.StandardGpuResources()
    config = faiss.GpuIndexIVFScalarQuantizerConfig()
    
    #The metric can be changed
    index = faiss.GpuIndexIVFScalarQuantizer(res, d, nlist,
    faiss.ScalarQuantizer.QT_4bit, faiss.METRIC_L2,config)
    
    index.train(xb)
    index.add(xb)
    distances, indices = index.search(xb, 5)
    ```
    

### **Inverted File and Product Quantization algorithm (IVFPQ)**

- This algorithm is a variation of the previous one.
- It uses the partitions, that are found by the IVF algorithm, and applies the Product Quantization
- Product Quantization, is when the number of centroids are drastically increased, by splitting each vector in others
- So, this algorithm, has the idea of the IVF algorithm, but adds the Product Quantization concept to perform the search
- This is an Approximate Nearest Neighbor methods
- This method is probabilistic

**Pros**

- The only method with sublinear space, great compression ratio (log(k) bits per vector.
- We can tune the parameters to change the accuracy/speed tradeoff.
- We can tune the parameters to change the space/accuracy tradeoff.
- Support batch queries.

**Cons**

- The exact nearest neighbor might be across the boundary to one of the neighboring cells.
- Can't incrementally add points to it.
- The exact nearest neighbor might be across the boundary to one of the neighboring cells.
    
    Link: [https://www.sciencedirect.com/science/article/abs/pii/S0020025522006806](https://www.sciencedirect.com/science/article/abs/pii/S0020025522006806)
    
- This algorithm is present in the cuML documentation, but cuML is linked with a unstable version of faiss, resulting in lots of erros
- So, in this test the Faiss version will be used
- ****************************************How to use in Python****************************************
    
    ```python
    import numpy as np
    import pdb
    
    import faiss
    
    print("load data")
    
    xb, xq, xt, gt = load_sift1M()
    nq, d = xb.shape
    nlist = 1000
    M = 4
    nbits = 8
    
    # we need only a StandardGpuResources per GPU
    res = faiss.StandardGpuResources()
    config = faiss.GpuIndexIVFPQConfig()
    
    #The metric can be changed
    index = faiss.GpuIndexIVFPQ(res, d, nlist, M, nbits,
                                      faiss.METRIC_L2, config)
    
    index.train(xb)
    index.add(xb)
    distances, indices = index.search(xb, 5)
    ```
    

## Evaluation Protocol

- To analysis the results of the kNN searchs the evaluation method that will be used is Recall@
- This method compares the first @  neighbors of each point
- Link de referência
- Furthermore, for the probabilistic algorithms( IVFFLAT, IVFPQ, IVFSQ) the search will be perfomed more than once, and the recall will be calculated more than once too
- This is done because a probability algorithm is based in probability and this means that one search may not reflect the power of it.

## Checking GPU-Usage

- To check GPU-Usage the nvidia-smi resource will be used while the test are going on
    
    
   ![Screenshot from 2022-12-16 10-48-19](https://user-images.githubusercontent.com/91696970/208112704-adece009-822a-4cbd-943b-57f4759f6857.png)



## Datasets

- To begin with, as the GPU that is being used is GTX …, that have 8 GB of memory, the datasets can’t be so huge, as the brute force algorithm is going to be tested too
- But, the datasets need to be huge enough to use the GPU
    
    ### Real datasets
    
    - The first real dataset that is going to be used in the tests is SIFT1M
    - Link para o dataset
    - The second real dataset that is going to be used in the tests is GIST1M
    - Link para o dataset
    
    ### Artificial datasets
    
- Mostrar quais datasets serão utilizados e diferenciar entre artificias e reais
- The datasets will be created using sklearn methods, that allows multi distributions in one dataset
- The parameters (n and dim) may be adjusted for the GPU capacity, but the idea is to test the for a n in range $1e6$ to $10e6$, and dim in range $2$ to $40$
- This values may not be the real one to be tested, because the GPU may no handle this ammount of data.
