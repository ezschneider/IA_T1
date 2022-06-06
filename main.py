from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy import stats
import numpy as np

# Função para organizar as medidas alvos
def scores_target(scores):
    st = [scores.mean(), scores.std()]
    inf, sup = stats.norm.interval(0.95, loc=st[0], scale=st[1]/np.sqrt(len(scores)))
    st = st + [inf] + [sup]
    return st

# Importando a base de dados Iris
from sklearn import datasets

iris = datasets.load_iris()
iris_x = iris.data
iris_y = iris.target   


# Importando os Classificadores ZeroR e Naive Bayes Gaussiano
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB

scalar = StandardScaler()
rkf = RepeatedStratifiedKFold(n_splits=10, n_repeats=3)

zR = DummyClassifier()
pipeline_zR = Pipeline([('transformer', scalar), ('estimator', zR)])

gNB = GaussianNB()
pipeline_gNB = Pipeline([('transformer', scalar), ('estimator', gNB)])

st_zR = scores_target(cross_val_score(pipeline_zR, iris_x, iris_y, scoring='accuracy', cv = rkf))
st_gNB = scores_target(cross_val_score(pipeline_gNB, iris_x, iris_y, scoring='accuracy', cv = rkf))

print("ZeroR Classifier")
print("\nMean Accuracy: %0.2f Standard Deviation: %0.2f" % (st_zR[0], st_zR[1]))
print ("Accuracy Confidence Interval (95%%): (%0.2f, %0.2f)\n" % 
       (st_zR[2], st_zR[3]))

print("Navy Base Classifier")
print("\nMean Accuracy: %0.2f Standard Deviation: %0.2f" % (st_gNB[0], st_gNB[1]))
print ("Accuracy Confidence Interval (95%%): (%0.2f, %0.2f)\n" % 
       (st_gNB[2], st_gNB[3]))