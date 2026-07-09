# 🩺 Sistema de Diagnóstico Preditivo para Diabetes (Machine Learning)

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Conclu%C3%ADdo-success?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)

## 📌 Descrição do Projeto
Em um ambiente hospitalar, o diagnóstico precoce e preciso de diabetes salva vidas e otimiza recursos. Este projeto é um sistema de **Machine Learning (Árvore de Decisão)** desenvolvido para atuar como um "segundo olhar" médico, triando pacientes com base em seus exames de sangue e histórico de saúde. 

O modelo foi projetado com um foco rigoroso na área da saúde, possuindo um *recall* de **95% para pacientes de alto risco**, garantindo que quase nenhum doente passe despercebido. Além disso, o sistema conta com uma esteira automatizada de auditoria de dados, que trata valores faltantes no prontuário utilizando imputação estatística (Média para exames, Moda para dados cadastrais), evitando a perda de histórico de pacientes.

---

## 🛠️ Índice
- [Demonstração Visual](#-demonstração-visual)
- [Pré-requisitos](#-pré-requisitos)
- [Instalação e Uso](#-instalação-e-uso)
- [Destaques da Arquitetura de Dados](#-destaques-da-arquitetura-de-dados)
- [Tecnologias Utilizadas](#-tecnologias-utilizadas)
- [Como Contribuir](#-como-contribuir)
- [Licença](#-licença)
- [Contato](#-contato)

---

## 📊 Demonstração Visual

O sistema gera automaticamente relatórios visuais essenciais para a explicabilidade da IA (*Explainable AI*) perante a equipe médica. 

* **Relação de Importância dos Exames:** Mostra quais fatores a IA considerou mais graves para o diagnóstico.
![Importância dos Exames](./grafico_importancia.png)

* **Análise Detalhada (Acertos e Erros):** Visão corporativa sobre a performance do sistema.
![Gráfico de Barras](./grafico_barras_previsao_diabetes.png)

* **Matriz de Confusão e Correlação:** O panorama estatístico e o "Raio-X" das variáveis do hospital.
![Matriz de Confusão](./mapa_calor.png) 
![Matriz de Correlação](./mapa_correlacao.png)

### Output do Terminal (Auditoria Médica em Tempo Real)
```text
--- AUDITORIA DE EXAMES (NÚMEROS) ---
Nenhum exame numérico nulo encontrado.

--- AUDITORIA DE CADASTRO (TEXTOS) ---
Nenhum dado de texto nulo encontrado.

              precision    recall  f1-score   support
 Baixo Risco       0.99      0.82      0.90     18292
  Alto Risco       0.32      0.95      0.48      1708
