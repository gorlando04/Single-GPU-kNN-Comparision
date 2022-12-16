#Importar as bibliotecas necessárias
import faiss
import numpy as np
from time import time   
import pandas as pd
from time import sleep

def bvecs_read(fname):
    a = np.fromfile(fname, dtype=np.int32, count=1)
    b = np.fromfile(fname, dtype=np.uint8)
    d = a[0]
    return b.reshape(-1, d + 4)[:, 4:].copy()

def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname):
    return ivecs_read(fname).view('float32')


#Reading vector
SIFT = fvecs_read('sift_base.fvecs')

def set_colors(rows,N):
    colors = np.zeros(N)
    
    cores = rows.shape[0]
    sample = rows.shape[1]
    
    for k in range(cores):
        for i in range(sample):
            if rows[k,i]:
                colors[i] = k + 1
    return colors

#Concatena as distribuições probabilísticas
def make_sample(data):
    
    
    sample = data[0]
    for i in range(1,len(data)):
        sample = np.concatenate((sample,data[i]))
    
    
    return sample

def make_colors(colors):
    
    sample = colors[0]
    max_c = max(colors[0])
    
    for i in range(1,len(colors)):
        colors[i] = colors[i] + max_c + 1   
        max_c = max(colors[i])
        sample = np.concatenate((sample,colors[i]))
    return sample

def biclust_dataset(N,dim):
    #Building make_bicluster dataset
    from sklearn.datasets import make_biclusters
    X0, rows,_ = make_biclusters(
    shape=(N, dim), n_clusters=2, noise=.4,minval=-12,maxval=10, shuffle=False, random_state=10)
    y0 = set_colors(rows,N) #Colors
    
    return X0,y0

def blobs_dataset(N,dim):
    #Building make_blobs dataset
    from sklearn.datasets import make_blobs
    X1, y1 = make_blobs(n_samples=N, centers=5, n_features=dim,
                   random_state=10,cluster_std=.6)
    return X1,y1

def normalize_dataset(data):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    norm_data = scaler.fit_transform(data)
    
    return norm_data

def get_artifical_db(N,dim):
    
    old_n = N
    N = N//2
    
    x0,y0 = biclust_dataset(N,dim)
    
    x1,y1 = blobs_dataset(N,dim)
    
    data = [x0,x1]
    colors = [y0,y1]
    
    sample = make_sample(data)
    col_list = make_colors(colors)
    
    #É preciso normalizar o conjunto de dados, visto que a distância utilizada é a euclidiana
    normalized_sample = normalize_dataset(sample)
    np.random.shuffle(normalized_sample)
    return normalized_sample,col_list

def create_dataset(N,dim):
    
    sample,col_list = get_artifical_db(N,dim)
    colors = col_list
    N = sample.shape[0]
    i0 = 0
    for i in range(N//2,len(colors),N):
        
        c_unique = colors[i0:i]
        c_out = colors[i:]
        
        unique = np.sort(pd.unique(c_unique))
        unique_out = np.sort(pd.unique(c_out))
        
        i0 = i
        
        for i in unique:
            if i in unique_out:
                print(f"O valor {i} esta na lista {unique_out}")
                exit()
      
    return sample.astype(np.float32),col_list
def recall(arr1,arr2,k):
    
    #Verificação da integridade
    if arr1.shape != arr2.shape:
        print("Impossível de fazer a avaliação, as arrays tem tamanho diferentes")
    elif arr1.shape[1] < k:
        print(f"Impossível de fazer o recall{k}, já que as array não tem {k} vizinhos")
    
    #Somatório dos k primeiros vizinhos positivos dividido por n*k
    acertos = 0
    
    n = arr1.shape[0]


    recall_value = (arr1[:,:k] == arr2[:,:k]).sum() / (float(n*k))
    
    return recall_value
    

def analysis_runtime(values):
    
    values = np.array(values)

    mean = values.mean()
  
    
    
    return mean


def brute_knn(data,k):
    #Uses euclidian distance
    
    _,d = data.shape
    
    # we need only a StandardGpuResources per GPU
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = 0

    t0 = time()
    index = faiss.GpuIndexFlatL2(res, d, flat_config)


    index.add(data)
    _, indices = index.search(data, k)
    
    tf = time() - t0
    return np.array(indices),tf


def IVFFlat_knn(data,k):

    n, d = data.shape
    nlist = 4000

    # we need only a StandardGpuResources per GPU
    res = faiss.StandardGpuResources()
    config = faiss.GpuIndexIVFFlatConfig()
    config.device = 0

    t0 = time()
    #The metric can be changed
    index = faiss.GpuIndexIVFFlat(res, d, nlist, faiss.METRIC_L2,config)
    if n > int(6e6):
        index.nprobe = 30
    else:
        index.nprobe = 40
    index.train(data)
    index.add(data)
    _, indices = index.search(data, k)
    
    tf = time() - t0
    return np.array(indices),tf


def IVFSQ_knn(data,k):

    n, d = data.shape
    nlist = 5000
    qtype = faiss.ScalarQuantizer.QT_4bit
    metric = faiss.METRIC_L2

    # we need only a StandardGpuResources per GPU
    res = faiss.StandardGpuResources()
    config = faiss.GpuIndexIVFScalarQuantizerConfig()
    config.device = 0
    
    t0 = time()
    #The metric can be changed
    index = faiss.GpuIndexIVFScalarQuantizer(res, d, nlist,
    qtype,faiss.METRIC_L2,True,config)
    if n > int(6e6):
        index.nprobe = 30
    else:
        index.nprobe = 40
    index.train(data)
    index.add(data)
    _, indices = index.search(data, k)
    tf = time() - t0
    
    return np.array(indices),tf



def IVFPQ_knn(data,k):

    n, d = data.shape


    # we need only a StandardGpuResources per GPU
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = 0

    index = faiss.GpuIndexFlatL2(res, d, flat_config)    

    M_list = [2,4,8,16,32,64]
    M = d
    count = len(M_list)-1
    while M % M_list[count] != 0:
        count -= 1
        if count == 0:
            break
    M = M_list[count]
    index = faiss.index_factory(d, f"IVF4096,PQ{M}")
    co = faiss.GpuClonerOptions()
    co.useFloat16 = True

    index = faiss.index_cpu_to_gpu(res, 0, index, co)


    t0 = time()
    # Create the index.

    if n > int(6e6):
        index.nprobe = 30
    else:
        index.nprobe = 40

    index.train(data)
    index.add(data)
    _, indices = index.search(data, k)
    tf = time() - t0
    
    return np.array(indices),tf


def write_df(df,index,name,method,dim,n_sample,time_knn,rec_value,k):
    df.loc[index, 'Name'] = name
    df.loc[index, 'Method'] = method
    df.loc[index, 'Dim'] = dim
    df.loc[index, 'N_sample'] = n_sample
    df.loc[index, 'Time kNN'] = time_knn
    df.loc[index, f'Recall@{k}'] = rec_value


metodos = {'deterministico':{'brute':brute_knn},
           'probabilistico':{'ivfflat': IVFFlat_knn,
                            'ivfpq': IVFPQ_knn,
                            'ivfsq':IVFSQ_knn}
}


index = 0

import sys
entrada = sys.argv
default_k, default_rec,default_sr = (10,10,1)

if len(entrada) == 4:
    default_k = int(entrada[1])
    default_rec = int(entrada[2])
    default_sr = int(entrada[3])
elif  len(entrada) == 3:
    default_k = int(entrada[1])
    default_rec = int(entrada[2])
elif len(entrada) == 2:
    default_k = int(entrada[1])
    

K = default_k
rec_k = default_rec
safe_repetitions = default_sr

print(f"K = {K}, Recall@ = {rec_k} and  safe repetitions = {safe_repetitions}")
dbs = {'SIFT1M':SIFT,
        'SK-1M-2d': create_dataset(int(1e6),2)[0],
        'SK-1M-10d':create_dataset(int(1e6),10)[0],
        'SK-1M-20d':create_dataset(int(1e6),20)[0],
        'SK-1M-40d':create_dataset(int(1e6),40)[0],
        'SK-2M-2d':create_dataset(int(2e6),2)[0],
        'SK-5M-2d':create_dataset(int(5e6),2)[0],
        'SK-10M-2d':create_dataset(int(10e6),2)[0],

      }

#Agora métodos diferentes que encontram os KNN em GPU serão testados
df_gpu = pd.DataFrame()

print("Começando a testagem")
for db in dbs:
        
    #Transform to a cudf df
    n_sample,dim = dbs[db].shape
    X = dbs[db]

    #Use all RAPIDS kNN methods
    brute_indices = None
    for c in metodos:
        for method in metodos[c]:
            reps = 1
            if c == 'probabilistico':
                reps = safe_repetitions

            if index == 0:
                #Warm up the GPU
                print("Warming up the GPU...")
                _,_ = metodos[c][method](X,K)
                print("GPU Ready to go...")
                sleep(5)
           

            index += 1

            #List if the are repetitions
            recall_list = list()
            time_knn_list = list()

            for i in range(reps):

                #Perform the knn search
                indices,time_knn = metodos[c][method](X,K)

                rec_value = '-'

                #Save the brute indices (exact result)
                if method == 'brute':
                    #Salva os indices exatos
                    brute_indices = indices.copy()

                    #p = pd.DataFrame(brute_indices)
                    #p.to_csv('indexes.csv', index=False)


                else:
                    #Recall calculation
                    rec_value = recall(brute_indices,indices,rec_k)
                    recall_list.append(rec_value)

                time_knn_list.append(time_knn)
                del indices
            #Sujeito a mudanças, aqui pode ser qualquer tipo de valor derivado das execuções
            if method != 'brute': 
                time_knn = analysis_runtime(time_knn_list)
                rec_value = analysis_runtime(recall_list)

            #Save the results in a dataframe
            write_df(df_gpu,index,db,method,dim,n_sample,time_knn,rec_value,rec_k)

            #Exibe o tempo
            print(f"Iteration -> {index} DB -> {db} Dim -> {dim} N -> {n_sample} Finished in {time_knn:.5} secs, method -> {method}, rep -> {reps}")            


df_gpu.to_csv('raw_data_gpu.csv', index=False)
print(df_gpu)