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

print('ZR:')
classification_report(cross_val_score(pipeline_zR, data_X, data_y, scoring='accuracy', cv=rkf))
print('GNB:')
classification_report(cross_val_score(pipeline_gNB, data_X, data_y, scoring='accuracy', cv=rkf))

# Parte II
# Importando os Classificadores KNN, KMC e AD
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.utils.validation import check_X_y

class KMeansClassifier():
    def __init__(self, cx=None, cy=None):
        super().__init__()
        self.cx = cx
        self.cy = cy

    def fit(self,x_train,y_train):
        x_train,y_train = check_X_y(x_train,y_train)

        counter = KMeans()
        self.__self_pred = max(counter,key = counter.get)

    def predict(self,x_test):
        (n,_) = x_test.shape
        n0 = floor(n*self.cx/100)
        n1 = floor(n*self.cy/100)
        nprd = n - (n0 + n1)        
        prd = [self.__self_pred]*nprd + [0]*n0 + [1]*n1
        return np.array(prd)

# Definindo os hiperparametros
parameters_KNN = {'estimator__n_neighbors':[1,3,5,7]}
parameters_AD = {'estimator__max_depth':[None,3,5,10]}
parameters_KMC = {'estimator__k':[1,3,5,7]}

kNN = KNeighborsClassifier()
pipeline_kNN = Pipeline([('transformer', scalar), ('estimator', kNN)])
p_KNN = GridSearchCV(pipeline_kNN, parameters_KNN, scoring='accuracy', cv=4)

aD = DecisionTreeClassifier()
pipeline_AD = Pipeline([('transformer', scalar), ('estimator', aD)])
p_AD = GridSearchCV(pipeline_AD, parameters_AD, scoring='accuracy', cv=4)

kMC = KMeansClassifier()
pipeline_kMC = Pipeline([('transformer', scalar), ('estimator', kMC)])
p_KMC = GridSearchCV(pipeline_kMC, parameters_KMC, scoring='accuracy', cv=4)

print('KNN:')
classification_report(cross_val_score(p_KNN, data_X, data_y, scoring='accuracy', cv=rkf))
print('AD:')
classification_report(cross_val_score(p_AD, data_X, data_y, scoring='accuracy', cv=rkf))
#print('KMC:')
#classification_report(cross_val_score(p_KMC, data_X, data_y, scoring='accuracy', cv = rkf))