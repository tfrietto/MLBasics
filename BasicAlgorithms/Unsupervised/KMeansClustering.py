import numpy
import sklearn
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn import metrics

# load data in and scale it down to reduce time
digits = load_digits()
data = scale(digits.data)

# assigns our data target
y = digits.target

# dynamic way to allocate how many k's, can also put hard value
k = len(numpy.unique(y))

# object decomposition to separate data
samples, features = data.shape

# scoring function from the sklearn website, could also make your own
def bench_k_means(estimator, name, data):
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.inertia_,
             metrics.homogeneity_score(y, estimator.labels_),
             metrics.completeness_score(y, estimator.labels_),
             metrics.v_measure_score(y, estimator.labels_),
             metrics.adjusted_rand_score(y, estimator.labels_),
             metrics.adjusted_mutual_info_score(y,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean')))

# creates classifier
clf = KMeans(n_clusters=k, init="random", n_init=10, max_iter=300)
bench_k_means(clf,1,data)
