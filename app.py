import streamlit as st
import pandas as pd
import joblib

# ==========================================
# 1. CONFIGURAÇÃO DA PÁGINA
# ==========================================
st.set_page_config(page_title="Passos Mágicos - Previsão de Risco", page_icon="🪄", layout="wide")
st.title("🪄 Previsão de Risco Educacional (2024)")
st.write("Identifique antecipadamente alunos com risco de aumento de defasagem ou queda de desempenho, com base no histórico de 2022 e 2023.")

# ==========================================
# 2. CARREGAR MODELO E DADOS
# ==========================================
@st.cache_resource
def carregar_modelo():
    try:
        return joblib.load('modelo_passos_magicos_vf.pkl')
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        return None

@st.cache_data
def carregar_dados_alunos():
    try:
        return pd.read_csv('dados_alunos_passos_magicos.csv')
    except Exception:
        return None

modelo = carregar_modelo()
df_alunos = carregar_dados_alunos()

colunas_esperadas = [
    'Defasagem_2022', 'IAA_2022', 'IDA_2022', 'IEG_2022', 'INDE_2022', 'IPP_2022', 'IPS_2022', 'IPV_2022',
    'Defasagem_2023', 'IAA_2023', 'IDA_2023', 'IEG_2023', 'INDE_2023', 'IPP_2023', 'IPS_2023', 'IPV_2023'
]

# ==========================================
# 3. CRIAR AS ABAS DA INTERFACE
# ==========================================
if modelo:
    tab1, tab2, tab3 = st.tabs(["👤 Previsão Individual", "📂 Previsão em Lote", "📊 Dashboard (Power BI)"])

    # ------------------------------------------
    # ABA 1: PREVISÃO INDIVIDUAL
    # ------------------------------------------
    with tab1:
        st.subheader("Análise de um único aluno")
        
        # Opção de escolha para o professor
        modo_entrada = st.radio(
            "Como deseja inserir os dados do aluno?", 
            ["🔍 Buscar Aluno na Base (Automático)", "✍️ Preenchimento Manual (Simulação)"], 
            horizontal=True
        )
        st.write("")

        # --- MODO AUTOMÁTICO (BUSCA NA BASE) ---
        if modo_entrada == "🔍 Buscar Aluno na Base (Automático)":
            if df_alunos is not None:
                # Criar uma lista bonita com Nome e RA
                df_alunos['Identificacao'] = df_alunos['Nome'] + " (RA: " + df_alunos['RA'].astype(str) + ")"
                lista_alunos = ["Selecione um aluno..."] + df_alunos['Identificacao'].tolist()
                
                aluno_selecionado = st.selectbox("Selecione o Aluno para análise:", lista_alunos)
                
                if aluno_selecionado != "Selecione um aluno...":
                    # Puxar os dados daquele aluno específico
                    dados_aluno = df_alunos[df_alunos['Identificacao'] == aluno_selecionado].iloc[0]
                    
                    
                    
                    
                    st.markdown("### 📈 Evolução do Aluno (2022 ➔ 2023)")
                    
                    # --- CÁLCULO EXATO DAS DIFERENÇAS ---
                    inde_22 = dados_aluno['INDE_2022']
                    inde_23 = dados_aluno['INDE_2023']
                    def_22 = dados_aluno['Defasagem_2022']
                    def_23 = dados_aluno['Defasagem_2023']
                    
                    diff_inde = inde_23 - inde_22
                    diff_def = def_23 - def_22
                    
                    # Lógica da Seta Neutra com o valor antigo (INDE)
                    if diff_inde > 0:
                        txt_inde = f"↗ +{diff_inde:.1f} (Em 2022 era {inde_22:.1f})"
                    elif diff_inde < 0:
                        txt_inde = f"↘ {diff_inde:.1f} (Em 2022 era {inde_22:.1f})"
                    else:
                        txt_inde = f"➔ Manteve (Em 2022 era {inde_22:.1f})"
                        
                    # --- TRADUÇÃO DA DEFASAGEM PARA TEXTO HUMANO (CORRIGIDO) ---
                    def traduz_defasagem(valor):
                        if valor < 0:  # Negativo = Atrasado
                            return f"{abs(valor):.0f} ano(s) atrasado" # abs() tira o sinal de menos para o texto ficar bonito
                        elif valor > 0: # Positivo = Adiantado
                            return f"{valor:.0f} ano(s) adiantado" 
                        else:
                            return "Na fase ideal"
                            
                    texto_def_23 = traduz_defasagem(def_23)
                    texto_def_22 = traduz_defasagem(def_22)
                        
                    # Lógica da Seta Neutra com o valor Defasagem)
                    # Se diff > 0, o número cresceu (ex: de -2 para -1), então MELHOROU
                    if diff_def > 0:
                        txt_def = f"↗ Melhorou (Recuperou {diff_def:.0f} ano(s) vs 2022)"
                    # Se diff < 0, o número caiu (ex: de 0 para -1), então PIOROU
                    elif diff_def < 0:
                        txt_def = f"↘ Piorou (Atrasou {abs(diff_def):.0f} ano(s) vs 2022)"
                    else:
                        txt_def = f"➔ Manteve (Em 2022: {texto_def_22})"

                    # --- CARDS NEUTROS E EXPLICATIVOS ---
                    c1, c2 = st.columns(2)
                    
                    with c1:
                        st.metric(label="INDE (Índice de Desenvolvimento Educacional)", value=f"{inde_23:.1f}")
                        st.caption(txt_inde) 
                        
                    with c2:
                        st.metric(label="Defasagem Escolar", value=texto_def_23)
                        st.caption(txt_def)
                    
                    st.write("") # Espaço para respirar
                    



                    # Botão para gerar a previsão
                    st.write("")
                    if st.button("🔮 Gerar Previsão para 2024", type="primary", use_container_width=True):
                        # Separar apenas os números para o modelo
                        X_aluno = pd.DataFrame([dados_aluno[colunas_esperadas].values], columns=colunas_esperadas)
                        
                        previsao = modelo.predict(X_aluno)[0]
                        probabilidade = modelo.predict_proba(X_aluno)[0][1] * 100
                        
                        st.divider()
                        if previsao == 1:
                            st.error(f"⚠️ **ALERTA DE RISCO DETECTADO PARA 2024!**")
                            st.write(f"O modelo indica uma probabilidade de **{probabilidade:.1f}%** de este aluno apresentar queda de INDE ou aumento de defasagem neste ano.")
                            st.write("Recomendação: Iniciar acompanhamento psicopedagógico focado e tutoria preventiva.")
                        else:
                            st.success(f"✅ **ALUNO COM TRAJETÓRIA SEGURA.**")
                            st.write(f"A probabilidade de risco é baixa (**{probabilidade:.1f}%**). O aluno apresenta resiliência académica.")

            else:
                st.warning("⚠️ O ficheiro 'dados_alunos_passos_magicos.csv' não foi encontrado. Por favor, utilize o preenchimento manual.")

        # --- MODO MANUAL ---
        else:
            col_2022, col_2023 = st.columns(2)
            
            with col_2022:
                st.markdown("### 📅 Dados de 2022")
                def_22 = st.number_input("Defasagem (2022)", min_value=-5, max_value=5, value=None, step=1, placeholder="Ex: 0")
                iaa_22 = st.number_input("IAA (2022)", min_value=0.0, max_value=10.0, value=None, step=0.1)
                ida_22 = st.number_input("IDA (2022)", min_value=0.0, max_value=10.0, value=None, step=0.1)
                ieg_22 = st.number_input("IEG (2022)", min_value=0.0, max_value=10.0, value=None, step=0.1)
                inde_22 = st.number_input("INDE (2022)", min_value=0.0, max_value=10.0, value=None, step=0.1)
                ipp_22 = st.number_input("IPP (2022)", min_value=0.0, max_value=10.0, value=None, step=0.1)
                ips_22 = st.number_input("IPS (2022)", min_value=0.0, max_value=10.0, value=None, step=0.1)
                ipv_22 = st.number_input("IPV (2022)", min_value=0.0, max_value=10.0, value=None, step=0.1)

            with col_2023:
                st.markdown("### 📅 Dados de 2023")
                def_23 = st.number_input("Defasagem (2023)", min_value=-5, max_value=5, value=None, step=1, placeholder="Ex: 0")
                iaa_23 = st.number_input("IAA (2023)", min_value=0.0, max_value=10.0, value=None, step=0.1)
                ida_23 = st.number_input("IDA (2023)", min_value=0.0, max_value=10.0, value=None, step=0.1)
                ieg_23 = st.number_input("IEG (2023)", min_value=0.0, max_value=10.0, value=None, step=0.1)
                inde_23 = st.number_input("INDE (2023)", min_value=0.0, max_value=10.0, value=None, step=0.1)
                ipp_23 = st.number_input("IPP (2023)", min_value=0.0, max_value=10.0, value=None, step=0.1)
                ips_23 = st.number_input("IPS (2023)", min_value=0.0, max_value=10.0, value=None, step=0.1)
                ipv_23 = st.number_input("IPV (2023)", min_value=0.0, max_value=10.0, value=None, step=0.1)

            st.write("")
            if st.button("🔮 Simular Risco do Aluno", use_container_width=True):
                inputs_lista = [def_22, iaa_22, ida_22, ieg_22, inde_22, ipp_22, ips_22, ipv_22, 
                                def_23, iaa_23, ida_23, ieg_23, inde_23, ipp_23, ips_23, ipv_23]
                
                if None in inputs_lista:
                    st.warning("⚠️ Por favor, preencha todos os indicadores de 2022 e 2023 para realizar a simulação.")
                else:
                    dados_aluno = pd.DataFrame([inputs_lista], columns=colunas_esperadas)
                    
                    previsao = modelo.predict(dados_aluno)[0]
                    probabilidade = modelo.predict_proba(dados_aluno)[0][1] * 100
                    
                    st.divider()
                    if previsao == 1:
                        st.error(f"⚠️ **ALERTA DE RISCO DETETADO!**")
                        st.write(f"O modelo indica uma probabilidade de **{probabilidade:.1f}%** de risco com base nestes dados fictícios.")
                    else:
                        st.success(f"✅ **CENÁRIO SEGURO.**")
                        st.write(f"A probabilidade de risco simulada é baixa (**{probabilidade:.1f}%**).")


    # ------------------------------------------
    # ABA 2: PREVISÃO EM LOTE
    # ------------------------------------------
    with tab2:
        st.subheader("Análise de múltiplos alunos via Planilha")
        st.write("Faça o upload de uma planilha (CSV ou Excel) contendo os indicadores de 2022 e 2023 da turma.")
        
        df_template = pd.DataFrame(columns=['RA', 'Nome'] + colunas_esperadas)
        csv_template = df_template.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="⬇️ Baixar Planilha Modelo",
            data=csv_template,
            file_name='template_passos_magicos.csv',
            mime='text/csv',
            help="Baixe esta planilha, preencha com os dados dos alunos e faça o upload abaixo."
        )
        
        st.divider()
        arquivo_upload = st.file_uploader("📂 Faça o upload da sua planilha preenchida", type=["csv", "xlsx"])
        
        if arquivo_upload is not None:
            try:
                if arquivo_upload.name.endswith('.csv'):
                    df_upload = pd.read_csv(arquivo_upload)
                else:
                    df_upload = pd.read_excel(arquivo_upload)
                
                st.write("Pré-visualização dos dados carregados:")
                st.dataframe(df_upload.head(3))
                
                colunas_faltantes = [col for col in colunas_esperadas if col not in df_upload.columns]
                
                if len(colunas_faltantes) > 0:
                    st.error(f"❌ Erro: A sua planilha não possui as colunas obrigatórias: {colunas_faltantes}")
                else:
                    if st.button("🚀 Processar Análise de Risco em Lote", type="primary"):
                        with st.spinner('A analisar padrões neurais...'):
                            X_lote = df_upload[colunas_esperadas]
                            X_lote = X_lote.fillna(X_lote.median())
                            
                            df_upload['Alerta_Risco'] = modelo.predict(X_lote)
                            df_upload['Probabilidade_Risco (%)'] = modelo.predict_proba(X_lote)[:, 1] * 100
                            df_upload['Probabilidade_Risco (%)'] = df_upload['Probabilidade_Risco (%)'].round(1)
                            
                            df_upload['Status'] = df_upload['Alerta_Risco'].apply(lambda x: '⚠️ EM RISCO' if x == 1 else '✅ SEGURO')
                            df_resultado = df_upload.sort_values(by='Probabilidade_Risco (%)', ascending=False)
                            
                            st.success("Análise concluída com sucesso!")
                            
                            colunas_tela = ['RA', 'Nome', 'Status', 'Probabilidade_Risco (%)'] if 'Nome' in df_upload.columns and 'RA' in df_upload.columns else ['Status', 'Probabilidade_Risco (%)']
                            st.dataframe(df_resultado[colunas_tela])
                            
                            csv_resultado = df_resultado.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="📥 Baixar Relatório Completo de Previsões (CSV)",
                                data=csv_resultado,
                                file_name='relatorio_risco_passos_magicos_2024.csv',
                                mime='text/csv',
                            )
            except Exception as e:
                st.error(f"Erro ao processar o ficheiro. Detalhes: {e}")

    # ------------------------------------------
    # ABA 3: DASHBOARD POWER BI
    # ------------------------------------------
    with tab3:
        st.subheader("📊 Painel Analítico de Indicadores")
        st.write("Explore os dados históricos e a evolução de todos os alunos da Associação Passos Mágicos.")
        
        power_bi_link = "https://app.powerbi.com/view?r=eyJrIjoiM2E2MDQ3NWUtNWNlMC00ZDgyLWFmZTUtZGY1NDZjYzhhYmQzIiwidCI6IjExZGJiZmUyLTg5YjgtNDU0OS1iZTEwLWNlYzM2NGU1OTU1MSIsImMiOjR9" 
        st.components.v1.iframe(power_bi_link, width=1000, height=600, scrolling=True)


# ==============================================================================
# RODAPÉ 
# ==============================================================================
    st.markdown("---") 

    col_info, col_dev = st.columns([2, 1])

    # --- Coluna da Esquerda: Sobre o Projeto ---
    with col_info:
        st.subheader("📌 Sobre o Projeto")
        st.markdown("""
        Este projeto foi desenvolvido para o **Datathon** da pós-graduação em Data Analytics (FIAP + Alura), utilizando dados reais da **Associação Passos Mágicos** — uma ONG que transforma a vida de crianças e jovens em vulnerabilidade social através da educação. 
        
        Minha missão foi analisar as dores do dia a dia da instituição e criar um modelo preditivo integrado a este *Data App*. O objetivo do sistema é prever de forma antecipada o risco de um aluno sofrer queda de desempenho ou aumento da defasagem escolar, permitindo uma ação preventiva da equipe.
        """)

    # --- Coluna da Direita: Desenvolvedora ---
    with col_dev:
        st.subheader("👩‍💻 Desenvolvido por")
        
        c_img, c_txt = st.columns([0.6, 2])
        
        with c_img:
            st.image("https://github.com/gesianne9.png", width=90) 
        
        with c_txt:
            st.markdown("""
            <div style='margin-top: 5px;'>
                <span style='font-size: 18px; font-weight: bold;'>Gesianne de Azevedo Ferreira</span>
                <br>
                <span style='font-size: 14px; color: rgba(250, 250, 250, 0.6);'>Cientista de Dados em Formação</span>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div style='margin-top: 10px;'>
                <a href='https://www.linkedin.com/in/gesianne-azevedo/'><img src='https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white' height='20'></a>
                <a href='https://github.com/gesianne9'><img src='https://img.shields.io/badge/GitHub-100000?style=flat&logo=github&logoColor=white' height='20'></a>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: grey; font-size: 12px;'>© Projeto Passos Mágicos - FIAP + Alura</div>", 
        unsafe_allow_html=True
    )