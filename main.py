from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from scipy import stats
import numpy as np

# Função para mostrar as medidas alvos
def classification_report(scores):
    print(f'Media: {scores.mean():.2f}, Desvio Padrao: {scores.std():.2f}')
    inf, sup = stats.norm.interval(0.95, loc=scores.mean(), 
                               scale=scores.std()/np.sqrt(len(scores)))
    print(f'Intervalo de confiança (95%): [{inf:.2f},{sup:.2f}]')

# Importando a base de dados Iris
from sklearn import datasets

iris = datasets.load_iris()
iris_x = iris.data
iris_y = iris.target   

# Parte I
# Importando os Classificadores ZeroR e Naive Bayes Gaussiano
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB

scalar = StandardScaler()
rkf = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=36851234)

zR = DummyClassifier()
pipeline_zR = Pipeline([('transformer', scalar), ('estimator', zR)])

gNB = GaussianNB()
pipeline_gNB = Pipeline([('transformer', scalar), ('estimator', gNB)])

print('ZR:')
classification_report(cross_val_score(pipeline_zR, iris_x, iris_y, scoring='accuracy', cv=rkf))
print('GNB:')
classification_report(cross_val_score(pipeline_gNB, iris_x, iris_y, scoring='accuracy', cv=rkf))

# Parte II
# Importando os Classificadores KNN, KMC e AD
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
#from sklearn.cluster import KMeans

# Definindo os hiperparametros
parameters_KNN = {'estimator__n_neighbors':[1,3,5,7]}
parameters_AD = {'estimator__max_deph':[None,3,5,10]}
parameters_KMC = {'estimator__k':[1,3,5,7]}

kNN = KNeighborsClassifier()
pipeline_kNN = Pipeline([('transformer', scalar), ('estimator', kNN)])
p_KNN = GridSearchCV(pipeline_kNN, parameters_KNN, scoring='accuracy', cv=4)

aD = DecisionTreeClassifier()
pipeline_AD = Pipeline([('transformer', scalar), ('estimator', aD)])
p_AD = GridSearchCV(pipeline_AD, parameters_AD, scoring='accuracy', cv=4)

# Implementar o KMC
#kMC = KMeans()
#pipeline_kMC = Pipeline([('transformer', scalar), ('estimator', kMC)])
#p_KMC = GridSearchCV(pipeline_kMC, parameters_KMC, scoring='accuracy', cv=4)

print('KNN:')
classification_report(cross_val_score(p_KNN, iris_x, iris_y, scoring='accuracy', cv=rkf))
print('AD:')
classification_report(cross_val_score(p_AD, iris_x, iris_y, scoring='accuracy', cv=rkf))
#print('KMC:')
#classification_report(cross_val_score(p_KMC, iris_x, iris_y, scoring='accuracy', cv = rkf))