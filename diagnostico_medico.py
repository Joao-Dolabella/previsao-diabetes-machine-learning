import pandas as pd
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler


#Configuracao global da fonte

#Altera a fonte de todo o grafico para uma familia sem serifa
plt.rcParams['font.family'] = 'sans-serif'
#Especifica quais fontes do sistema o Matplotlib deve tentar usar
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'Dejavu Sans']

#abrindo o arquivo e salvando os dados temporariamente - ja organiza em tabela e nao precisa de binarizar pois o arquivo ja conta com resultado binarizado
tabela = pd.read_csv("diabetes_prediction_dataset.csv")

#limpeza dos dados do arquivo csv
def limpar_dados(tabela_suja: pd.DataFrame) -> pd.DataFrame:
  tabela_limpa = tabela_suja.dropna(axis = 'index', how = 'any')  #axis = 0 e axis = 'index' remove a coluna. axis = 1 e axis = 'columns' remove a coluna
  return tabela_limpa

tabela_limpa = limpar_dados(tabela) 

#conversao para matriz X e vetor y
def conversao_matriz_vetor(tabela_incompleta: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
  matriz = tabela_incompleta[['gender', 'age', 'hypertension', 'heart-disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']]
  vetor = tabela_incompleta['diabetes']
  return matriz, vetor

X, y = conversao_matriz_vetor(tabela_limpa)

X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

#escalonando as colunas igualmente (fazendo ficarem na mesma escala matematica de variancia)
def escalonando(X_treino_escalonamento: pd.DataFrame, X_teste_escalonamento: pd.DataFrane) -> tuple[pd.DataFrame, pd.DataFrame]:
  
  if sp.issparse(X_treino_escalonamento) or sp.issparse(X_teste_escalonamento):
    escalonador = StandardScaler(with_mean=False)
  else:
    escalonador = StandardScaler()
  
  escalonador.fit_transform(X_treino_escalonamento)
  escalonador.transform(X_teste_escalonamento)
  return X_treino_escalonamento

X_treino_escalonado = escalonando(X_treino, X_teste)

#instanciamento e treinamento do modelo
def treinar_modelo(X_exercicio: pd.DataFrame, y_exercicio: pd.Series) -> DecisionTreeClassifier:
  modelo = DecisionTreeClassifier()
  modelo.fit(X_exercicio, y_exercicio)
  return modelo

modelo_treinado = treinar_modelo(X_treino, y_treino)

#realizamento da "prova" pela IA utilizando predict
def avaliar_modelo(modelo_em_avaliacao: DecisionTreeClassifier) -> np.ndarray:
  previsoes = modelo_em_avaliacao.predict(X_teste)
  return previsoes

deducoes_modelo = avaliar_modelo(modelo_treinado)

#precisao, recall e f1 - scores que avaliam o quanto o modelo eh preciso com falsos positivos, falsos negativos, verdadeiros positivos e negativos
precision = precision_score(y_teste, deducoes_modelo)
recall = recall_score(y_teste, deducoes_modelo)
f1= f1_score(y_teste, deducoes_modelo)

#visualizacao grafica do desempenho do modelo em passos (socorro, alguem me salva, ta dando muito trabalho :sob:)

#passo 1 = extrair os valores da matriz de confusao - tn = true negative, fp = false positive, fn = false negative e tp = true positive
tn, fp, fn, tp = confusion_matrix(y_teste, deducoes_modelo).ravel()

def grafico_barras():
    #calcular os totais reais por classe para descobrir as porcentagens 
  total_classe_A = tn + fp    #Total de amostras que sao realmente Classe A
  total_classe_B = fn + tp    #Total de amostras que sao realmente Classe B

  #Criar as strings formatadas combinando: "Valor (Porcentagem)"
  texto_tn = f"{tn}\n({(tn / total_classe_A) * 100:.1f})"
  texto_fp = f"{fp}\n({(fp / total_classe_A) * 100:.1f}%)"
  texto_fn = f"{fn}\n({(fn / total_classe_B) * 100:.1f}%)"
  texto_tp = f"{tp}\n({(tp / total_classe_B) * 100:.1f}%)"

  #Estruturar os dados para os eixos do graficos
  classes = ['Classe A (Baixo risco)', 'Classe B (Fraude)']
  acertos = [tn, tp]
  erros = [fp, fn]

  #Listas com os textos personalizados que vao substituir os numeros brutos nas barras
  rotulos_acertos = [texto_tn, texto_tp]
  rotulos_erros = [texto_fp, texto_fn]

  #Criar a plotagem com Matplotlib
  fig, ax = plt.subplots(figsize=(8, 5.5))   #altura aumentada para caber o texto de duas linhas

  #Remover bordas desnecessarias
  for spine in ['top', 'right', 'left']:
    ax.spines[spine].set_visible(False)
  
  #Desenhar as barras empilhadas
  bar_acertos = ax.bar(classes, acertos, label='Acertos (Previsão Correta)', color="#07db18", width=0.45)
  bar_erros = ax.bar(classes, erros, bottom=acertos, label='Erros (Previsão Incorreta)', color='#e71d36', width=0.45)
  
  #Adicionar os rotulos DUPLOS customizados dentro de cada bloco
  ax.bar_label(bar_acertos, labels=rotulos_acertos, label_type='center', color='white', weight='bold', fontsize=11)
  ax.bar_label(bar_erros, labels=rotulos_erros, label_type='center', color='white', weight='bold', fontsize=11)

  ## Customização final de títulos e legendas
  ax.set_title('Análise Detalhada de Acertos e Erros do Modelo', fontsize=16, pad=25, weight='bold', color='#1d3557')
  ax.tick_params(axis='x', labelsize=12, colors='#1d3557')
  ax.get_yaxis().set_visible(False) # Oculta o eixo Y padrão
  ax.legend(loc='upper right', frameon=False, fontsize=10)
  
  fig.savefig('grafico_barras_previsao_diabetes.pdf', format='pdf')

  plt.tight_layout()
  plt.show()

#def mapa_calor():

  #disp = ConfusionMatrixDisplay.from_predictions(
    #y_teste,
    #deducoes_modelo,
    #display_labels = ['Baixo risco', 'Alto risco'],
    #cmap = plt.cm.Reds
  #)

  #plt.show()

print(classification_report(y_teste, deducoes_modelo, target_names=['Baixo risco','Alto risco']))