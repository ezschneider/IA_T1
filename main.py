from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from scipy import stats
import pandas as pd
import numpy as np

table_result = pd.DataFrame(columns=['Método', 'Média', 'Desvio Padrão', 'Limite Inferior', 'Limite Superior'])

# Função para mostrar as medidas alvos
def classification_report(scores):
    print(f'Media: {scores.mean():.2f}, Desvio Padrao: {scores.std():.2f}')
    inf, sup = stats.norm.interval(0.95, loc=scores.mean(), 
                               scale=scores.std()/np.sqrt(len(scores)))
    print(f'Intervalo de confiança (95%): [{inf:.2f},{sup:.2f}]')

# Função para adicionar as medidas alvos na tabela de scores
def add_result(met, df, scores):
    inf, sup = stats.norm.interval(0.95, loc=scores.mean(), 
                               scale=scores.std()/np.sqrt(len(scores)))
    new_row = pd.DataFrame([[met, round(scores.mean(), 2), round(scores.std(), 2), round(inf, 2), round(sup, 2)]],
                            columns=['Método', 'Média', 'Desvio Padrão', 'Limite Inferior', 'Limite Superior'])

    df = pd.concat([df, new_row])
    return df
    

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

table_result = add_result('ZR', table_result, cross_val_score(pipeline_zR, data_X, data_y, scoring='accuracy', cv=rkf))
#classification_report(cross_val_score(pipeline_zR, data_X, data_y, scoring='accuracy', cv=rkf))
table_result = add_result('NBG', table_result, cross_val_score(pipeline_gNB, data_X, data_y, scoring='accuracy', cv=rkf))
#classification_report(cross_val_score(pipeline_gNB, data_X, data_y, scoring='accuracy', cv=rkf))

# Parte II
# Importando os Classificadores KNN, KMC e AD
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.utils.validation import check_X_y


# class KMeansClassifier():
#    def __init__(self, cx=None, cy=None):
#        super().__init__()
#        self.cx = cx
#        self.cy = cy

#    def fit(self,x_train,y_train):
#        x_train,y_train = check_X_y(x_train,y_train)

#        counter = KMeans()
#        self.__self_pred = max(counter,key = counter.get)

#    def predict(self,x_test):
       

# Definindo os hiperparametros
parameters_KNN = {'estimator__n_neighbors':[1,3,5,7]}
parameters_AD = {'estimator__max_depth':[None,3,5,10]}
#parameters_KMC = {'estimator__k':[1,3,5,7]}

#kMC = KMeansClassifier()
#pipeline_kMC = Pipeline([('transformer', scalar), ('estimator', kMC)])
#p_KMC = GridSearchCV(pipeline_kMC, parameters_KMC, scoring='accuracy', cv=4)

kNN = KNeighborsClassifier()
pipeline_kNN = Pipeline([('transformer', scalar), ('estimator', kNN)])
p_KNN = GridSearchCV(pipeline_kNN, parameters_KNN, scoring='accuracy', cv=4)

aD = DecisionTreeClassifier()
pipeline_AD = Pipeline([('transformer', scalar), ('estimator', aD)])
p_AD = GridSearchCV(pipeline_AD, parameters_AD, scoring='accuracy', cv=4)

#table_result = add_result('KMC', table_result, cross_val_score(p_AD, data_X, data_y, scoring='accuracy', cv=rkf))
#classification_report(cross_val_score(p_KMC, data_X, data_y, scoring='accuracy', cv = rkf))
table_result = add_result('KNN', table_result, cross_val_score(p_KNN, data_X, data_y, scoring='accuracy', cv=rkf))
#classification_report(cross_val_score(p_KNN, data_X, data_y, scoring='accuracy', cv=rkf))
table_result = add_result('AD', table_result, cross_val_score(p_AD, data_X, data_y, scoring='accuracy', cv=rkf))
#classification_report(cross_val_score(p_AD, data_X, data_y, scoring='accuracy', cv=rkf))

print(table_result)