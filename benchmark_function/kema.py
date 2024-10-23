from sklearn.cluster import KMeans
import numpy as np

# サンプルデータの生成
X = np.random.rand(100, 2)

# クラスタ数を10に指定
kmeans = KMeans(n_clusters=10, n_init=10)
kmeans.fit(X)

# クラスタ数が指定通りか確認
unique_labels = np.unique(kmeans.labels_)
print("クラスタ数（実際の）:", len(unique_labels))