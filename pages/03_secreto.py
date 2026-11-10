import streamlit as st
import os

# Efeito visual que chove balões na tela quando a página carrega
st.balloons() 

st.markdown("""
### Pabéns! 
Quem explora todos os detalhes do material sempre acha as melhores coisas. Aqui está a recompensa:
""")

# Coloque uma foto engraçada da equipe ou um meme na pasta images
caminho_foto_secreta = os.path.join(os.getcwd(), "images", "rhyron.jpg") 

if os.path.exists(caminho_foto_secreta):
    # O use_container_width garante que a foto respeite as margens da tela
    st.image(caminho_foto_secreta, caption="Rhyron e seu abacaxi no dia do lançamento do Erippus", use_container_width=True)
else:
    st.warning("⚠️ Salve uma imagem chamada 'secreto.jpg' na sua pasta 'images/'.")