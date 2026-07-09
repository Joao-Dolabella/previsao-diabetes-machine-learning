import streamlit as st
import pandas as pd
import joblib
import time

# 1. Configuração da Página (Deve ser a primeira linha do Streamlit)
st.set_page_config(page_title="Sistema de Triagem - Diabetes", page_icon="🏥", layout="wide")

# 2. Carregando os Modelos
@st.cache_resource # Isso faz o site carregar muito mais rápido na nuvem
def carregar_modelos():
    modelo = joblib.load('modelo_diagnostico_diabetes.joblib')
    escalonador = joblib.load('escalonador_diabetes.joblib')
    return modelo, escalonador

modelo, escalonador = carregar_modelos()

# 3. Cabeçalho Principal
st.title("🏥 Sistema Inteligente de Triagem Preditiva")
st.markdown("""
Este sistema utiliza um modelo de **Machine Learning (Random Forest)** para analisar dados vitais e prever o risco de diabetes. 
*Por favor, preencha o prontuário no menu lateral.*
""")
st.divider()

# 4. Menu Lateral (Sidebar) para o Formulário
st.sidebar.header("📝 Prontuário do Paciente")

with st.sidebar:
    st.subheader("Exames Vitais")
    age = st.number_input("Idade", min_value=0, max_value=120, value=50, help="Idade do paciente em anos completos.")
    bmi = st.number_input("IMC (Índice de Massa Corporal)", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
    hba1c = st.number_input("Nível de HbA1c (%)", min_value=3.0, max_value=15.0, value=5.5, step=0.1)
    glicose = st.number_input("Nível de Glicose (mg/dL)", min_value=50, max_value=300, value=100)
    
    st.markdown("---")
    
    st.subheader("Fatores de Risco")
    hypertension = st.selectbox("Hipertensão?", ["Não", "Sim"])
    heart_disease = st.selectbox("Doença Cardíaca?", ["Não", "Sim"])
    
    st.markdown("---")
    
    st.subheader("Dados Cadastrais")
    gender = st.selectbox("Gênero", ["Feminino", "Masculino", "Outro"])
    smoking = st.selectbox("Histórico de Fumo", ["Nunca fumou", "Ex-fumante", "Fumante atual", "Sempre fumou", "Não fuma atualmente", "Sem informação"])
    
    # Botão no menu lateral
    analisar_btn = st.button("Gerar Diagnóstico Preditivo", type="primary", use_container_width=True)

# 5. Processamento e Exibição dos Resultados (Centro da tela)
if analisar_btn:
    # Animação de carregamento (dá um ar de processamento corporativo)
    with st.spinner("Analisando padrões vitais do paciente..."):
        time.sleep(1.5) # Pausa dramática de 1.5 segundos
        
        # Tradução (Igual ao anterior)
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
        
        df_paciente = pd.DataFrame([dados_paciente])
        df_escalonado = escalonador.transform(df_paciente)
        
        # A Mágica Nova: predict_proba traz a porcentagem
        probabilidades = modelo.predict_proba(df_escalonado)[0]
        prob_doente = probabilidades[1] * 100 # Pegamos a chance de ser da classe 1 (Alto Risco)
        
        # Exibição do Resultado
        st.subheader("📊 Resultado da Análise")
        
        # Colunas para exibir os números de forma elegante (Métricas)
        col1, col2, col3 = st.columns(3)
        col1.metric("Glicose Analisada", f"{glicose} mg/dL")
        col2.metric("HbA1c Analisada", f"{hba1c}%")
        col3.metric("Risco Calculado pela IA", f"{prob_doente:.1f}%")
        
        st.markdown("---")
        
        # Veredito Final
        if prob_doente >= 50.0:
            st.error(f"🚨 **ALERTA CLÍNICO:** O modelo detectou um risco de **{prob_doente:.1f}%** para diabetes. Recomenda-se encaminhamento imediato para exames laboratoriais complementares.")
        else:
            st.success(f"✅ **BAIXO RISCO:** O paciente apresenta apenas **{prob_doente:.1f}%** de indicativos de diabetes. Mantenha rotina preventiva anual.")

# 6. Rodapé / Disclaimer Legal
st.markdown("---")
st.caption("⚠️ **Aviso Legal:** Este sistema é uma prova de conceito (PoC) para fins educacionais e de portfólio. Não substitui, em hipótese alguma, o diagnóstico de um profissional de saúde qualificado.")