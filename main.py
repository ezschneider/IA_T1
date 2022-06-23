from cmath import inf
from random import random
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from scipy import stats
import pandas as pd
import numpy as np

table_result_measure = pd.DataFrame(columns=['Método', 'Média', 'Desvio Padrão', 'Limite Inferior', 'Limite Superior'])
table_result_pvalue = pd.DataFrame(columns=[''])

# Função para mostrar as medidas alvos
def classification_report(scores):
    print(f'Media: {scores.mean():.2f}, Desvio Padrao: {scores.std():.2f}')
    inf, sup = stats.norm.interval(0.95, loc=scores.mean(), 
                               scale=scores.std()/np.sqrt(len(scores)))
    print(f'Intervalo de confiança (95%): [{inf:.2f},{sup:.2f}]')

# Função para adicionar as medidas alvos na tabela de scores
def add_result_measure(met, df, scores):
    inf, sup = stats.norm.interval(0.95, loc=scores.mean(), 
                               scale=scores.std()/np.sqrt(len(scores)))
    new_row = pd.DataFrame([[met, round(scores.mean(), 2), round(scores.std(), 2), round(inf, 2), round(sup, 2)]],
                            columns=['Método', 'Média', 'Desvio Padrão', 'Limite Inferior', 'Limite Superior'])
    df = pd.concat([df, new_row])
    return df

# Função para adicionar as medidas alvos na tabela de p-value
def table_htest(list_scores):
    from scipy.stats import ttest_rel, wilcoxon
    list_estimators = ['ZR', 'NBG', 'KNN', 'AD'] #inluir kmc depois

    for row in range(len(list_estimators)):
        for col in range(len(list_estimators)):
            if(row == col):
                print(list_estimators[row], end = "\t")
            else:
                if(col > row):
                    s,p = ttest_rel(list_scores[row],list_scores[col])
                    print(round(p, 6), end = "\t")
                else:
                    s,p = wilcoxon(list_scores[row],list_scores[col])
                    print(round(p, 6), end = "\t")
        print ("\n")

# Importando a base de dados Iris
from sklearn import datasets

data = datasets.load_breast_cancer()
data_X = data.data
data_y = data.target   

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

zr_scores = cross_val_score(pipeline_zR, data_X, data_y, scoring='accuracy', cv=rkf)
table_result_measure = add_result_measure('ZR', table_result_measure, cross_val_score(pipeline_zR, data_X, data_y, scoring='accuracy', cv=rkf))
#classification_report(cross_val_score(pipeline_zR, data_X, data_y, scoring='accuracy', cv=rkf))
nbg_scores = cross_val_score(pipeline_gNB, data_X, data_y, scoring='accuracy', cv=rkf)
table_result_measure = add_result_measure('NBG', table_result_measure, cross_val_score(pipeline_gNB, data_X, data_y, scoring='accuracy', cv=rkf))
#classification_report(cross_val_score(pipeline_gNB, data_X, data_y, scoring='accuracy', cv=rkf))

# Parte II
# Importando os Classificadores KNN, KMC e AD
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y

class KMCClassifier(BaseEstimator):
    def __init__(self, k=1):
        super().__init__()
        self.k = k
        self.cent = []

    def fit(self, X_train, y_train):
        X_train, y_train = check_X_y(X_train, y_train)
        for classe in range(len(np.unique(y_train))):
            kmeans = KMeans(n_clusters=self.k)
            kmeans.fit(X_train [y_train == classe], classe)
            self.cent.append((kmeans.cluster_centers_, classe))

    def predict(self, X_test):
        list_labels = []
        for X_element in X_test:
            min_dist = float('inf')
            for list_cent in self.cent:
                for cent in list_cent[0]:
                    dist = np.linalg.norm(cent-X_element)
                    if min_dist > dist:
                        min_dist = dist
                        list_labels.append(list_cent[1])
        return list_labels

# Definindo os hiperparametros
parameters_KMC = {'estimator__k':[1,3,5,7]}
parameters_KNN = {'estimator__n_neighbors':[1,3,5,7]}
parameters_AD = {'estimator__max_depth':[None,3,5,10]}

kMC = KMCClassifier()
pipeline_kMC = Pipeline([('transformer', scalar), ('estimator', kMC)])
p_KMC = GridSearchCV(pipeline_kMC, parameters_KMC, scoring='accuracy', cv=4)

kNN = KNeighborsClassifier()
pipeline_kNN = Pipeline([('transformer', scalar), ('estimator', kNN)])
p_KNN = GridSearchCV(pipeline_kNN, parameters_KNN, scoring='accuracy', cv=4)

aD = DecisionTreeClassifier()
pipeline_AD = Pipeline([('transformer', scalar), ('estimator', aD)])
p_AD = GridSearchCV(pipeline_AD, parameters_AD, scoring='accuracy', cv=4)

kmc_scores = cross_val_score(p_KMC, data_X, data_y, scoring='accuracy', cv=rkf)
table_result_measure = add_result_measure('KMC', table_result_measure, kmc_scores)
#classification_report(cross_val_score(p_KMC, data_X, data_y, scoring='accuracy', cv = rkf))
knn_scores = cross_val_score(p_KNN, data_X, data_y, scoring='accuracy', cv=rkf)
table_result_measure = add_result_measure('KNN', table_result_measure, knn_scores)
#classification_report(cross_val_score(p_KNN, data_X, data_y, scoring='accuracy', cv=rkf))
ad_scores = cross_val_score(p_AD, data_X, data_y, scoring='accuracy', cv=rkf)
table_result_measure = add_result_measure('AD', table_result_measure, ad_scores)
#classification_report(cross_val_score(p_AD, data_X, data_y, scoring='accuracy', cv=rkf))

table_result_measure.reset_index(drop=True, inplace=True)
# print(table_result_measure)

list = [zr_scores, nbg_scores, knn_scores, ad_scores]
# table_htest(list)