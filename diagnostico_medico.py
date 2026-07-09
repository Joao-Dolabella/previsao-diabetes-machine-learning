import pandas as pd
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.ensemble import RandomForestClassifier

#Configuracao global da fonte

#Altera a fonte de todo o grafico para uma familia sem serifa
plt.rcParams['font.family'] = 'sans-serif'

#Especifica quais fontes do sistema o Matplotlib deve tentar usar
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'Dejavu Sans']

#abrindo o arquivo e salvando os dados temporariamente - ja organiza em tabela e nao precisa de binarizar pois o arquivo ja conta com resultado binarizado
tabela = pd.read_csv("diabetes_prediction_dataset.csv")

#audita e corrige os do arquivo csv, limpando a tabela e preenchendo os dados vazios com a media
def auditar_e_corrigir_dados(tabela_suja: pd.DataFrame) -> pd.DataFrame:
  tabela_copia = tabela_suja.copy() 
  
  # 1. Separar os grupos de colunas
  colunas_numericas = tabela_copia.select_dtypes(include='number').columns
  colunas_texto = tabela_copia.select_dtypes(exclude='number').columns
  
  # 2. Fazer o Raio-X isolado para cada grupo
  erros_numericos = tabela_copia[colunas_numericas].isna().sum()
  erros_texto = tabela_copia[colunas_texto].isna().sum()
  
  # 3. Criar as máscaras (filtrar apenas o que é maior que zero)
  nulos_numericos = erros_numericos[erros_numericos > 0]
  nulos_texto = erros_texto[erros_texto > 0]
  
  # --- RELATÓRIO E CURA DOS NÚMEROS ---
  print("--- AUDITORIA DE EXAMES (NÚMEROS) ---")
  if nulos_numericos.empty:
    print("Nenhum exame numérico nulo encontrado.\n")
  else:
    print(f"---ALERTA DE AUDITORIA---\nValores nulos encontrados:\nPreenchendo-os com a MÉDIA do hospital:\n{nulos_numericos}\n")    
  for coluna in colunas_numericas:
    media_coluna = tabela_copia[coluna].mean()
    tabela_copia[coluna] = tabela_copia[coluna].fillna(media_coluna)
      
  # --- RELATÓRIO E CURA DOS TEXTOS ---
  print("--- AUDITORIA DE CADASTRO (TEXTOS) ---")
  if nulos_texto.empty:
    print("Nenhum dado de texto nulo encontrado.\n")
  else:
    print(f"---ALERTA DE AUDITORIA---\nTextos nulos encontrados\nPreenchendo-os com a MODA (mais comum) do hospital:\n{nulos_texto}\n")
    
  for coluna in colunas_texto:
    moda_coluna = tabela_copia[coluna].mode()[0]
    tabela_copia[coluna] = tabela_copia[coluna].fillna(moda_coluna)

  return tabela_copia

tabela_limpa = auditar_e_corrigir_dados(tabela) 

#traducao de string para binario
def string_para_binario(tabela_a_binarizar: pd.DataFrame) -> pd.DataFrame:
  tabela_apos_binarizacao = pd.get_dummies(tabela_a_binarizar, prefix=None, columns=['gender','smoking_history'], dtype=int)

  return tabela_apos_binarizacao

tabela_binarizada = string_para_binario(tabela_limpa)

#conversao para matriz X e vetor y
def conversao_matriz_vetor(tabela_incompleta: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
  matriz = tabela_incompleta[['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level',
       'blood_glucose_level', 'gender_Female', 'gender_Male',
       'gender_Other', 'smoking_history_No Info', 'smoking_history_current',
       'smoking_history_ever', 'smoking_history_former',
       'smoking_history_never', 'smoking_history_not current']]
  vetor = tabela_incompleta['diabetes']

  return matriz, vetor

X, y = conversao_matriz_vetor(tabela_binarizada)

X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

#escalonando as colunas igualmente (fazendo ficarem na mesma escala matematica de variancia)
def escalonando(X_treino_escalonamento: pd.DataFrame, X_teste_escalonamento: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
  
  if sp.issparse(X_treino_escalonamento) or sp.issparse(X_teste_escalonamento):
    escalonador = StandardScaler(with_mean=False)
  else:
    escalonador = StandardScaler()
  
  X_treino_escalonamento= escalonador.fit_transform(X_treino_escalonamento)
  X_teste_escalonamento= escalonador.transform(X_teste_escalonamento)

  return X_treino_escalonamento, X_teste_escalonamento, escalonador

X_treino_escalonado, X_teste_escalonado, escalonador_treinado = escalonando(X_treino, X_teste)

#instanciamento e treinamento do modelo
def treinar_modelo(X_exercicio: pd.DataFrame, y_exercicio: pd.Series) -> DecisionTreeClassifier:
  modelo = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=42)
  modelo.fit(X_exercicio, y_exercicio)
  return modelo

modelo_treinado = treinar_modelo(X_treino_escalonado, y_treino)

#realizamento da "prova" pela IA utilizando predict
def avaliar_modelo(modelo_em_avaliacao: DecisionTreeClassifier, X_teste_pronto) -> np.ndarray:
  previsoes = modelo_em_avaliacao.predict(X_teste_pronto)
  return previsoes

deducoes_modelo = avaliar_modelo(modelo_treinado, X_teste_escalonado)

#precisao, recall e f1 - scores que avaliam o quanto o modelo eh preciso com falsos positivos, falsos negativos, verdadeiros positivos e negativos
precision = precision_score(y_teste, deducoes_modelo)
recall = recall_score(y_teste, deducoes_modelo)
f1= f1_score(y_teste, deducoes_modelo)

#visualizacao grafica do desempenho do modelo em passos (socorro, alguem me salva, ta dando muito trabalho :sob:)

#passo 1 = extrair os valores da matriz de confusao - tn = true negative, fp = false positive, fn = false negative e tp = true positive
tn, fp, fn, tp = confusion_matrix(y_teste, deducoes_modelo).ravel()

#calcular os totais reais por classe para descobrir as porcentagens 
total_classe_A = tn + fp    #Total de amostras que sao realmente Classe A
total_classe_B = fn + tp    #Total de amostras que sao realmente Classe B

#Criar as strings formatadas combinando: "Valor (Porcentagem)"
texto_tn = f"{tn}\n({(tn / total_classe_A) * 100:.1f})"
texto_fp = f"{fp}\n({(fp / total_classe_A) * 100:.1f}%)"
texto_fn = f"{fn}\n({(fn / total_classe_B) * 100:.1f}%)"
texto_tp = f"{tp}\n({(tp / total_classe_B) * 100:.1f}%)"

#Criar matrizes para os valores e as anotacoes a fim de serem usadas pelo seaborn. Valores para saber onde eh mais escuro ou claro
matriz_valores = [[tn,fp], [fn,tp]]
matriz_anotacoes = [[texto_tn, texto_fp], [texto_fn, texto_tp]]

def grafico_barras(total_classe_A: int, total_classe_B: int, texto_tn: str, texto_fp: str, texto_fn: str, texto_tp: str):
  #Estruturar os dados para os eixos do graficos
  classes = ['Classe A (Baixo Risco)', 'Classe B (Alto Risco)']
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
  
  # O label_type='edge' joga o texto pro topo, e o padding=3 dá um respiro para não encostar na linha
  ax.bar_label(bar_erros, labels=rotulos_erros, label_type='edge', padding=3, color='black', weight='bold', fontsize=11)

  ## Customização final de títulos e legendas
  ax.set_title('Análise Detalhada de Acertos e Erros do Modelo', fontsize=16, pad=25, weight='bold', color='#1d3557')
  ax.tick_params(axis='x', labelsize=12, colors='#1d3557')
  ax.get_yaxis().set_visible(False) # Oculta o eixo Y padrão
  ax.legend(loc='upper right', frameon=False, fontsize=10)
  
  fig.savefig('grafico_barras_previsao_diabetes.png', dpi=300, bbox_inches='tight')

  plt.tight_layout()

def grafico_importancia(modelo_treinado, X):
  plt.figure(figsize=(10, 6))
  nomes_colunas = X.columns
  importancias = modelo_treinado.feature_importances_
  
  tabela_importancia= pd.DataFrame({
    'Exame': nomes_colunas,
    'Importância': importancias
  }).sort_values(by='Importância', ascending=False)

  sns.barplot(data= tabela_importancia, x='Importância', y='Exame', hue='Exame', legend=False, palette='magma')

  plt.title('Relação Importância dos Exames', pad=20, fontsize=14, weight='bold')
  plt.tight_layout()
  plt.savefig('grafico_importancia.png', dpi=300, bbox_inches='tight')
  plt.show()

def mapa_calor():
  plt.figure(figsize=(8, 6))
  sns.heatmap(data= matriz_valores, annot= matriz_anotacoes, fmt="", cmap='Reds', xticklabels=['Baixo Risco', 'Alto Risco'], yticklabels=['Baixo Risco', 'Alto Risco'])
  
  plt.xlabel('Previsão da Máquina', fontsize=12, weight='bold')
  plt.ylabel('Diagnóstico Real', fontsize=12, weight='bold')
  plt.title('Matriz de Confusão do Diagnóstico', pad=20, fontsize=14, weight='bold')
  plt.tight_layout()
  plt.savefig('mapa_calor.png', dpi=300, bbox_inches='tight')
  plt.show()

def mapa_correlacao():
  plt.figure(figsize=(12, 8))
  correlacao = tabela_binarizada.corr()
  sns.heatmap(correlacao, annot=True, cmap='coolwarm', fmt=".2f")
  plt.title('Matriz de Correlação do Diagnóstico', pad=20, fontsize=14, weight='bold')
  plt.tight_layout()
  plt.savefig('mapa_correlacao.png', dpi=300, bbox_inches='tight')
  plt.show()

print(classification_report(y_teste, deducoes_modelo, target_names=['Baixo Risco','Alto Risco']))

grafico_barras(total_classe_A, total_classe_B, texto_tn, texto_fp, texto_fn, texto_tp)
grafico_importancia(modelo_treinado,X)

mapa_calor()
mapa_correlacao()

joblib.dump(modelo_treinado, 'modelo_diagnostico_diabetes.joblib')
joblib.dump(escalonador_treinado, 'escalonador_diabetes.joblib')