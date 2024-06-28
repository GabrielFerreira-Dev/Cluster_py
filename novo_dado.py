from pickle import load
# Simular uma nova instância
nova_instancia = [[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 1]]

# Normalizar a nova instância
normalizador = load(open('C:\\Users\\gabri\\PycharmProjects\\clusterizador_heart\\normalizador\\normalizador_heart.pkl', 'rb'))
nova_instancia_normalizada = normalizador.transform(nova_instancia)

# Carregar o modelo de clustering treinado
heart_cluster_model = load(open('C:\\Users\\gabri\\PycharmProjects\\clusterizador_heart\\normalizador\\heart_kmeans_model.pkl', 'rb'))

# Determinar a qual cluster a nova instância pertence
cluster_predito = heart_cluster_model.predict(nova_instancia_normalizada)[0]

# Descrever o cluster de forma compreensível
centroides = heart_cluster_model.cluster_centers_
descricao_cluster_normalizado = centroides[cluster_predito]

# Desnormalizar os valores do centróide
descricao_cluster = normalizador.inverse_transform([descricao_cluster_normalizado])[0]

# Apresentar a descrição do cluster
colunas = ['age', 'sex', 'chest pain type', 'resting bp s', 'cholesterol',
           'fasting blood sugar', 'resting ecg', 'max heart rate',
           'exercise angina', 'oldpeak', 'ST slope', 'target']

print(f"A nova instância foi classificada no cluster: {cluster_predito}")
print("Descrição do cluster (centroides desnormalizados):")
for coluna, valor in zip(colunas, descricao_cluster):
    print(f"{coluna}: {valor:.4f}")