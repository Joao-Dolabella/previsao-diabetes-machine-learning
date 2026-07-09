import streamlit as st
import pandas as pd
import joblib

# 1. Carregando o "Cérebro" e a "Régua"
modelo = joblib.load('modelo_diagnostico_diabetes.joblib')
escalonador = joblib.load('escalonador_diabetes.joblib')

# 2. Desenhando a Tela da Aplicação
st.set_page_config(page_title="Triagem Médica", page_icon="🩺", layout="centered")
st.title("🩺 IA Médica - Previsão de Diabetes")
st.write("Preencha os dados do paciente abaixo para realizar a triagem automática.")

st.markdown("---")

# 3. Criando as colunas do formulário
col1, col2 = st.columns(2)

with col1:
    st.subheader("Exames Vitais")
    age = st.number_input("Idade", min_value=0, max_value=120, value=50)
    bmi = st.number_input("IMC", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
    hba1c = st.number_input("Nível de HbA1c", min_value=3.0, max_value=15.0, value=5.5, step=0.1)
    glicose = st.number_input("Nível de Glicose", min_value=50, max_value=300, value=100)

with col2:
    st.subheader("Fatores de Risco e Cadastro")
    gender = st.selectbox("Gênero", ["Feminino", "Masculino", "Outro"])
    hypertension = st.selectbox("Hipertensão?", ["Não", "Sim"])
    heart_disease = st.selectbox("Doença Cardíaca?", ["Não", "Sim"])
    smoking = st.selectbox("Histórico de Fumo", ["Nunca fumou", "Ex-fumante", "Fumante atual", "Sempre fumou", "Não fuma atualmente", "Sem informação"])

st.markdown("---")

# 4. O Botão de Previsão
if st.button("Gerar Diagnóstico Preditivo", type="primary"):
    
    # 4.1 Traduzindo os textos do formulário para os "zeros e uns" que a máquina entende
    dados_paciente = {
        'age': age,
        'hypertension': 1 if hypertension == "Sim" else 0,
        'heart_disease': 1 if heart_disease == "Sim" else 0,
        'bmi': bmi,
        'HbA1c_level': hba1c,
        'blood_glucose_level': glicose,
        'gender_Female': 1 if gender == "Feminino" else 0,
        'gender_Male': 1 if gender == "Masculino" else 0,
        'gender_Other': 1 if gender == "Outro" else 0,
        'smoking_history_No Info': 1 if smoking == "Sem informação" else 0,
        'smoking_history_current': 1 if smoking == "Fumante atual" else 0,
        'smoking_history_ever': 1 if smoking == "Sempre fumou" else 0,
        'smoking_history_former': 1 if smoking == "Ex-fumante" else 0,
        'smoking_history_never': 1 if smoking == "Nunca fumou" else 0,
        'smoking_history_not current': 1 if smoking == "Não fuma atualmente" else 0
    }
    
    # 4.2 Convertendo para DataFrame
    df_paciente = pd.DataFrame([dados_paciente])
    
    # 4.3 Aplicando o Escalonador (A régua)
    df_escalonado = escalonador.transform(df_paciente)
    
    # 4.4 Realizando a Previsão
    resultado = modelo.predict(df_escalonado)
    
    # 5. Exibindo o Resultado na Tela
    if resultado[0] == 1:
        st.error("🚨 **ALERTA: Paciente classificado como ALTO RISCO para Diabetes.** Recomenda-se avaliação médica imediata.")
    else:
        st.success("✅ **BAIXO RISCO:** O paciente não apresenta indicativos graves de diabetes no momento.")