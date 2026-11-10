import streamlit as st
import os
import base64
from PIL import Image

# Caminho seguro para a sua imagem de ícone
caminho_icone = os.path.join(os.getcwd(), "images", "15.png")

# Tenta carregar a imagem. Se não achar, usa o foguete de fallback.
if os.path.exists(caminho_icone):
    icone_aba = Image.open(caminho_icone)
else:
    icone_aba = "🚀"

# Configuração global da página
st.set_page_config(
    page_title="Curso Python & Dados",
    page_icon=icone_aba, 
    layout="wide"
)

# ---------------------------------------------------------
# 1. CONFIGURAÇÃO DAS PÁGINAS E ROTEAMENTO (Oculto)
# ---------------------------------------------------------
home = st.Page("pages/00_home.py", title="Início", icon="🏠")
dia1 = st.Page("pages/01_dia_1.py", title="Dia 1: Fundamentos e Dados", icon="🐍")
dia2 = st.Page("pages/02_dia_2.py", title="Dia 2: Aeroespacial e IA", icon="✈️")
secreto = st.Page("pages/03_secreto.py", title="Confidencial", icon="🤫", url_path="secreto")

# Ao invés de deixar o Streamlit desenhar o menu automaticamente no topo, 
# nós o colocamos como position="hidden". Ele ainda gerencia a troca de páginas.
pg = st.navigation([home, dia1, dia2, secreto], position="hidden")


# ---------------------------------------------------------
# 2. TOPO DA BARRA LATERAL (Foto da Equipe)
# ---------------------------------------------------------
def carregar_imagem_redonda(caminho_imagem):
    if os.path.exists(caminho_imagem):
        with open(caminho_imagem, "rb") as img_file:
            b64_string = base64.b64encode(img_file.read()).decode()
        return f'''
            <div style="display: flex; justify-content: center; margin-bottom: 20px; margin-top: 10px;">
                <img src="data:image/jpeg;base64,{b64_string}" 
                     style="width: 140px; height: 140px; border-radius: 50%; object-fit: cover; border: 3px solid #ff4b4b; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
            </div>
        '''
    return ""

caminho_equipe = os.path.join(os.getcwd(), "images", "1.png")
html_equipe = carregar_imagem_redonda(caminho_equipe)

if html_equipe:
    st.sidebar.markdown(html_equipe, unsafe_allow_html=True)
else:
    st.sidebar.warning("⚠️ Adicione 'equipe.jpeg' na pasta images")


# ---------------------------------------------------------
# 3. MEIO DA BARRA LATERAL (Menu de Navegação Manual)
# ---------------------------------------------------------
#st.sidebar.markdown("### 📚 Navegação")

# Recriamos os botões de navegação na ordem que quisermos. 
# O page_link detecta a página atual sozinho e a destaca.
st.sidebar.page_link(home, label="Início", icon="🏠")
st.sidebar.page_link(dia1, label="Dia 1: Fundamentos", icon="🐍")
st.sidebar.page_link(dia2, label="Dia 2: Aeroespacial e IA", icon="✈️")

# ---------------------------------------------------------
# 4. ESPAÇADOR INVISÍVEL PARA EMPURRAR O RODAPÉ
# ---------------------------------------------------------
# Isso cria um bloco vazio ocupando 35% da altura da tela, empurrando o resto para baixo
#st.sidebar.markdown("<div style='height: 20vh;'></div>", unsafe_allow_html=True)
st.sidebar.markdown("<div style='margin-top: 90px;'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# 5. INFERIOR DA BARRA LATERAL (Logos de Patrocínio/Apoio)
# ---------------------------------------------------------
st.sidebar.markdown("---") # Linha divisória fina

# Logo do Evento
caminho_evento = os.path.join(os.getcwd(), "images", "evento.png")
if os.path.exists(caminho_evento):
    st.sidebar.image(caminho_evento, use_container_width=True)
else:
    st.sidebar.caption("🖼️ *Logo do Evento*")

st.sidebar.write("") # Espaço em branco para respirar

# Logos da UFRN e ECT Lado a lado
col1, col2 = st.sidebar.columns(2)

caminho_ufrn = os.path.join(os.getcwd(), "images", "ufrn.png")
if os.path.exists(caminho_ufrn):
    col1.image(caminho_ufrn, use_container_width=True)
else:
    col1.caption("🖼️ *UFRN*")
    
caminho_ect = os.path.join(os.getcwd(), "images", "ect.png")
if os.path.exists(caminho_ect):
    col2.image(caminho_ect, use_container_width=True)
else:
    col2.caption("🖼️ *ECT*")


# ---------------------------------------------------------
# 6. EXECUÇÃO DO ROTEADOR
# ---------------------------------------------------------
# Só agora mandamos o Streamlit rodar a página que o usuário escolheu.
pg.run()