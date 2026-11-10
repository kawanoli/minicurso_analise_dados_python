import streamlit as st
from componentes import contact_style, sobre_link
import base64
import os

# ---------------------------------------------------------
# BANNER FULL-WIDTH (De ponta a ponta)
# ---------------------------------------------------------
def carregar_banner(caminho_imagem):
    if os.path.exists(caminho_imagem):
        with open(caminho_imagem, "rb") as img_file:
            b64_string = base64.b64encode(img_file.read()).decode()
            
        st.markdown(f"""
            <style>
                .banner-container {{
                    width: 100%; /* Respeita as margens laterais nativas do Streamlit */
                    margin-top: -20px; /* Puxa levemente para perto do topo */
                    margin-bottom: 30px;
                }}
                .banner-container img {{
                    width: 100%;
                    height: 300px; /* Altura do banner */
                    object-fit: cover; /* Evita que a imagem estique */
                    border-radius: 20px; /* Deixa as bordas arredondadas */
                    box-shadow: 0 8px 16px rgba(0,0,0,0.15); /* Sombra 3D elegante */
                }}
            </style>
            <div class="banner-container">
                <img src="data:image/jpeg;base64,{b64_string}" alt="Banner">
            </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Imagem do banner não encontrada.")

# Chamando a função (coloque o nome da sua imagem aqui)
caminho_banner = os.path.join(os.getcwd(), "images", "banner.JPG") # <-- ajuste a extensão/nome

def sobre_mim():
    st.markdown("## 👋 Sobre mim")
    
    # Colunas para layout
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Imagem de perfil
        st.image("images/kawan.jpg", use_container_width=True)
    
    with col2:
        # Informações básicas
        st.markdown("""
        ### **Kawan Oliveira**  
        _Estudante de Engenharia da Computação_  
        UFRN, Departamento de Engenharia da Computação e Automação  
        
        ---
        """)
        
        st.markdown("""
        Olá 👋! Meu nome é Kawan, tenho 23 anos e sou um estudante de Engenharia da Computação na UFRN, e atualmente, membro do setor de eletrônica daqui do PRD

        - Técnico em Eletrotécnica pelo IFRN Central
        - Membro do PRD no setor de Eletrônica desde 2024.2
        - Ex Monitor de Lógica de Programação por 1 ano e meio
        - Ex Membro do URA, onde pude participar de eventos de divulgação científica
        - Apaixonado por engenharia e fascinado por veículos de engenharia (foguetes, carros, aviões, etc)
        - Minhas áreas de interesse são sistemas embarcados e visão computacional (e, claro, qualquer coisa correlata)
        - Atualmente faço parte da equipe de desenvolvimento do PotyraSat

        ---

        🚀🚀 Atualmente membro do setor de eletrônica da :rainbow[Potiguar Rocket Design (PRD)] 
        
        """)
        
        col_a, col_b, col_c = st.columns([1, 2, 1])
        with col_b:
            sobre_link()
        
        st.markdown("""
        ---
        Sinta-se à vontade para entrar em contato se tiver alguma oportunidade interessante ou sugestões! 😉
        ---
        """)
        
        col_d, col_e, col_f = st.columns([1, 2, 1])
        with col_d:
            st.markdown("""**Conecte-se comigo:** """)
        with col_e:
            contact_style()

# --- CONTEÚDO PRINCIPAL DA HOMEPAGE ---

st.title("🚀 Minicurso: Introdução à Análise de Dados com Python")
st.subheader("Da Coleta à Inteligência Artificial com foco no Setor Aeroespacial")

st.markdown("---")

st.markdown("""
### Bem-vindo(a)!
Este material interativo foi desenhado para um intensivo de **2 dias**. Vamos levar você do zero na programação até a construção de modelos preditivos e análises complexas usadas na engenharia aeroespacial.

Utilize o **menu lateral** para navegar entre os dias do curso.
""")

st.markdown("### 🗓️ Cronograma do Curso")

col1, col2 = st.columns(2)

with col1:
    st.info("""
    ### 🐍 Dia 1: A Base e Fundamentos
    **Setup e Python Básico**
    - Instalação e Ambiente
    - Sintaxe e Lógica
    - *Projeto: Aircraft Performance Calculator*
    
    **O Ciclo do Dado**
    - Ecossistema (Pandas e NumPy)
    - Data I/O e Tratamento
    - Visualização Básica (Matplotlib/Seaborn)
    """)
    
with col2:
    st.warning("""
    ### ✈️ Dia 2: Aplicações e Modelagem
    **Domínio Aeroespacial**
    - Formatos Específicos (HDF5)
    - Bibliotecas Setoriais (Poliastro)
    - Modelagem Matemática Aplicada
    
    **Inteligência e Interatividade**
    - Dashboards Interativos (Plotly)
    - Machine Learning (Scikit-learn)
    - Overview de Deep Learning (TensorFlow/NLTK)
    """)

st.markdown("---")
st.markdown("")
carregar_banner(caminho_banner)
st.markdown("---")

# Chama a função sobre mim no final da página
sobre_mim()

st.markdown("---")
# Link HTML disfarçado com CSS forçado (!important) para anular o estilo nativo do Streamlit
st.markdown("""
    <style>
        /* Aplicamos o mesmo estilo para o estado normal e o hover */
        a.easter-egg-caption, a.easter-egg-caption:hover, a.easter-egg-caption:active {
            text-decoration: none !important; 
            border-bottom: none !important; 
            color: rgba(128, 128, 128, 0.8) !important; /* Sempre cinza */
            font-size: 14px !important; 
            cursor: default !important; /* Mouse continua como seta/texto, NUNCA vira mãozinha */
            display: block; 
            background-color: transparent !important;
        }
    </style>
    
    <a href="secreto" target="_self" class="easter-egg-caption">
        Desenvolvido de forma interativa. Explore os gráficos e ferramentas ao longo das páginas.
    </a>
""", unsafe_allow_html=True)