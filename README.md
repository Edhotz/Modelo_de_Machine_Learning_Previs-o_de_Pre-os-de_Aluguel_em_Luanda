
# Modelo de Previsão de Preços de Aluguel em Luanda

Este guia fornece um exemplo completo de como treinar um modelo de regressão linear para prever preços de aluguel de propriedades em Luanda, Angola, com base em características da propriedade e sua localização.

## Dependências

Certifique-se de ter as seguintes bibliotecas Python instaladas:
- pandas
- scikit-learn
- numpy

Você pode instalá-las usando pip:

```bash
pip install pandas scikit-learn numpy
```

## Código

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Carregar os dados
df = pd.read_csv('caminho/para/o/arquivo/housing_luanda_atualizado.csv')  # Atualize com o caminho correto do arquivo

# Codificando a variável 'Localização' com OneHotEncoder
encoder = OneHotEncoder(sparse=False)
localizacao_encoded = encoder.fit_transform(df[['Localização']])

# Criando um DataFrame a partir das localizações codificadas
localizacao_encoded_df = pd.DataFrame(localizacao_encoded, columns=encoder.get_feature_names_out(['Localização']))

# Concatenando os novos dados codificados com os dados originais, excluindo a coluna 'Localização' original
df_encoded = pd.concat([df.drop(['ID', 'Tipo', 'Localização', 'Preço (USD)'], axis=1), localizacao_encoded_df], axis=1)

# Variável dependente
y = df['Preço (USD)']

# Dividindo os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(df_encoded, y, test_size=0.2, random_state=42)

# Treinando o modelo de regressão linear
model = LinearRegression()
model.fit(X_train, y_train)

# Predições e avaliação do modelo
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)

# Analisando os coeficientes das variáveis, incluindo as localizações, para identificar quais oferecem melhor valor
coeficientes = pd.DataFrame(model.coef_, df_encoded.columns, columns=['Coeficiente']).sort_values(by='Coeficiente')

print(f"RMSE: {rmse}")
print("Coeficientes do modelo:")
print(coeficientes)
```

## Instruções

1. Substitua `'caminho/para/o/arquivo/housing_luanda_atualizado.csv'` pelo caminho correto onde seu arquivo CSV está armazenado.
2. Execute o script Python para treinar o modelo e avaliar seu desempenho.
3. Analise os coeficientes impressos pelo modelo para entender o impacto das diferentes características e localizações nos preços de aluguel em Luanda.

## Conclusão

Este exemplo demonstra como preparar seus dados, treinar um modelo de regressão linear e analisar os resultados para tomar decisões informadas sobre o mercado imobiliário em Luanda.
