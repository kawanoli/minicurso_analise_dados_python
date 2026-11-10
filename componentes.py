import streamlit as st

#sidebar_style
def sidebar_style(secoes: list):
    st.sidebar.markdown("""
        <style>
        .custom-nav-container {
            background-color: #2f2f2f;
            padding: 16px;
            border-radius: 10px;
            border: 1px solid #444;
        }
        .custom-nav-title {
            color: white;
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 12px;
        }
        .custom-nav-link {
            color: white !important;
            text-decoration: none !important;
            font-weight: 500;
            display: block;
            margin-bottom: 8px;
        }
        .custom-nav-link:hover {
            color: #dddddd !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # Construir o conteúdo da caixa com título e links
    links_html = '<div class="custom-nav-container">'
    links_html += '<div class="custom-nav-title">Navegação</div>'
    for texto, id_ in secoes:
        links_html += f'<a class="custom-nav-link" href="#{id_}">{texto}</a>'
    links_html += '</div>'

    # Renderizar tudo de uma vez
    st.sidebar.markdown(links_html, unsafe_allow_html=True)

def contact_style():
    # Links sociais
    st.markdown("""
        <style>
        .button-container {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin-top: 10px;
        }
        .button-container a {
            text-decoration: none;
            padding: 8px 16px;
            background-color: #2f2f2f;
            color: white;
            border-radius: 5px;
            font-size: 14px;
            transition: background-color 0.3s ease;
        }
        .button-container a:hover {
            background-color: #03045E;
        }
        </style>
 
        <div class="button-container">
            <a href="https://www.linkedin.com/in/kawan-oliveira-187427291" target="_blank">LinkedIn</a>
            <a href="https://github.com/kawanoli" target="_blank">GitHub</a>
            <a href="https://instagram.com/kaw_yyy" target="_blank">Instagram</a>
            <a href="mailto:kawan.oliveira.712@ufrn.edu.br">Email</a>
        </div>
        """, unsafe_allow_html=True)
    
def sobre_link():
    # Links sociais
    st.markdown("""
        <style>
        .button-container {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin-top: 10px;
        }
        .button-container a {
            text-decoration: none;
            padding: 8px 16px;
            background-color: #2f2f2f;
            color: white;
            border-radius: 5px;
            font-size: 14px;
            transition: background-color 0.3s ease;
        }
        .button-container a:hover {
            background-color: #03045E;
        }
        </style>

        <div class="button-container">
            <a href="https://www.instagram.com/potiguarrocket/" target="_blank">Siga o PRD no Instagram</a>
        </div>
        """, unsafe_allow_html=True)

def remove_espaco_sup():
    # Reduz o espaçamento superior da página
    st.markdown(
        """
        <style>
            .block-container {
                padding-top: 1rem;  /* Valor padrão é ~6rem, diminua conforme quiser */
            }
        </style>
        """,
        unsafe_allow_html=True
    )