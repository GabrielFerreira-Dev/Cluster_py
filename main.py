import warnings
import pandas as pd
from sklearn import preprocessing
from pickle import dump
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, silhouette_score
import matplotlib.pyplot as plt
from yellowbrick.cluster import KElbowVisualizer

warnings.filterwarnings("ignore")

# Carregar os dados
dados = pd.read_csv('C:\\Users\\gabri\\PycharmProjects\\clusterizador_heart\\dados\\heart_statlog_cleveland_hungary_final.csv', sep=',', encoding='latin1')

print("\nInformações gerais sobre o DataFrame:")
print(dados.columns)

# Selecionar atributos
dados_atributos = dados[['age', 'sex', 'chest pain type', 'resting bp s', 'cholesterol',
       'fasting blood sugar', 'resting ecg', 'max heart rate',
       'exercise angina', 'oldpeak', 'ST slope', 'target']]

# Construir o pipeline de pré-processamento e clusterização
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', preprocessing.MinMaxScaler()),
    ('kmeans', KMeans(random_state=42))
])

# Determinar o número ideal de clusters usando o Elbow Method
dados_atributos_preprocessed = pipeline.named_steps['scaler'].fit_transform(
    pipeline.named_steps['imputer'].fit_transform(dados_atributos)
)
modelo_kmeans = KMeans(random_state=42)
visualizer = KElbowVisualizer(modelo_kmeans, k=(2, 20))

visualizer.fit(dados_atributos_preprocessed)
visualizer.show()

# Número ideal de clusters
numero_ideal_clusters = visualizer.elbow_value_

# Definir o grid de hiperparâmetros a serem otimizados
param_grid = {
    'kmeans__n_clusters': [numero_ideal_clusters],
    'kmeans__init': ['k-means++', 'random'],
    'kmeans__n_init': [10, 20, 30],
    'kmeans__max_iter': [300, 600, 900]
}

# Definir o scorer personalizado
def silhouette_scorer(estimator, X):
    labels = estimator.named_steps['kmeans'].predict(X)
    return silhouette_score(X, labels)

scorer = make_scorer(silhouette_scorer)

# Configurar o GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring=scorer)

# Realizar a busca pelos melhores hiperparâmetros
grid_search.fit(dados_atributos)

# Exibir os melhores hiperparâmetros encontrados
print("Melhores hiperparâmetros encontrados:")
print(grid_search.best_params_)

# Treinar o pipeline com os melhores hiperparâmetros
best_pipeline = grid_search.best_estimator_
best_pipeline.fit(dados_atributos)

# Salvar o pipeline treinado
dump(best_pipeline, open('C:\\Users\\gabri\\PycharmProjects\\clusterizador_heart\\normalizador\\heart_kmeans_pipeline.pkl', 'wb'))
