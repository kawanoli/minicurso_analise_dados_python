"""Dia 1 – Fundamentos e o Ciclo do Dado (minicurso Python para Aeroespacial)

Execute com:  streamlit run dia1.py
"""
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

try:  # dentro de st.navigation o app.py já define a configuração da página
    st.set_page_config(page_title="Dia 1 – Fundamentos e Ciclo do Dado", page_icon="🐍", layout="wide")
except st.errors.StreamlitAPIException:
    pass

G0 = 9.80665  # m/s²
KT_EM_MS = 0.514444  # 1 nó = 0,514444 m/s


# =====================================================================
# FUNÇÕES AUXILIARES (no curso: "refatorar para um módulo .py")
# =====================================================================
def atmosfera_isa(h):
    """Atmosfera Padrão Internacional (ISA) até 20 km. Retorna T [K], p [Pa], rho [kg/m³]."""
    h = np.asarray(h, dtype=float)
    T = np.where(h <= 11000, 288.15 - 0.0065 * h, 216.65)
    p = np.where(
        h <= 11000,
        101325 * (T / 288.15) ** 5.2559,
        22632.06 * np.exp(-G0 * (h - 11000) / (287.058 * 216.65)),
    )
    rho = p / (287.058 * T)
    return T, p, rho


def v_estol(massa_kg, area_m2, rho, cl_max, fator_carga=1.0):
    """Velocidade de estol [m/s]. fator_carga = n (1 em voo reto e nivelado)."""
    peso = massa_kg * G0
    return np.sqrt(2 * peso * fator_carga / (rho * area_m2 * cl_max))


# =====================================================================
# CABEÇALHO
# =====================================================================
st.title("🐍 Dia 1: Fundamentos e o Ciclo do Dado")
st.markdown(
    "Base de Python para engenharia e primeira volta completa no **ciclo da análise de dados**. "
    "Cada aba termina com uma **🎯 Atividade**. Coleta, tratamento e visualização ficam para o Dia 2."
)

abas = st.tabs(["🧭 Visão geral", "🛠️ Setup e Python básico", "🔢 NumPy e Pandas", "✈️ Aircraft Simulator"])

# ========= ABAS =========
with abas[0]:
    st.header("O ciclo da análise de dados")
    st.graphviz_chart(
        """
        digraph {
          rankdir=LR;
          node [shape=box, style="rounded,filled", fillcolor="#E8F1FB", fontname="Helvetica"];
          "1. Coleta" -> "2. Tratamento" -> "3. Modelagem" -> "4. Visualização" -> "Decisão";
          "Decisão" -> "1. Coleta" [style=dashed, label=" novas perguntas"];
        }
        """
    )
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📅 Dia 1")
        st.markdown(
            "- Apresentação das bibliotecas e do ambiente\n"
            "- Sintaxe Python aplicada à engenharia\n"
            "- **Coleta**: arquivos, APIs, raspagem, bancos, sensores, datasets\n"
            "- **Tratamento e filtragem** com Pandas\n"
            "- **Visualização** com Matplotlib e Seaborn"
        )
    with c2:
        st.subheader("📅 Dia 2")
        st.markdown(
            "- Data I/O e formatos (HDF5, Parquet)\n"
            "- Bibliotecas aeroespaciais\n"
            "- **Modelagem** matemática + visualizações interativas (Plotly)\n"
            "- statsmodels, scikit-learn, TensorFlow, NLTK\n"
            "- Mini-projeto integrador"
        )
    st.info(
        "💡 **Reatividade do Streamlit:** a cada interação o script inteiro roda de novo, de cima para baixo. "
        "É por isso que os dados são gerados dentro de funções com `@st.cache_data`."
    )

with abas[1]:
    st.header("1. Setup e ambiente")
    st.markdown(
        "Python em engenharia e dados depende de um **ecossistema**: interpretador, ambiente virtual, "
        "bibliotecas e um editor (VS Code ou Jupyter)."
    )
    st.code(
        "python -m venv .venv\n"
        "source .venv/bin/activate      # Windows: .venv\\Scripts\\activate\n"
        "pip install -r requirements.txt\n"
        "streamlit run dia1.py",
        language="bash",
    )

    st.subheader("Apresentação das bibliotecas")
    st.dataframe(
        pd.DataFrame(
            [
                ("NumPy", "Arrays e cálculo vetorizado", "Coleta/Tratamento", "Dia 1"),
                ("Pandas", "Tabelas (DataFrame), I/O, filtragem", "Coleta/Tratamento", "Dia 1"),
                ("Matplotlib / Seaborn", "Gráficos estáticos (relatórios)", "Visualização", "Dia 1"),
                ("Plotly", "Gráficos interativos (dashboards)", "Visualização", "Dia 2"),
                ("h5py", "Arquivos HDF5 (telemetria pesada)", "Coleta (I/O)", "Dia 2"),
                ("SciPy", "Ajuste de curvas, otimização, ODEs", "Modelagem", "Dia 2"),
                ("statsmodels", "Modelos estatísticos com inferência", "Modelagem", "Dia 2"),
                ("scikit-learn", "Machine Learning clássico", "Modelagem", "Dia 2"),
                ("TensorFlow / Keras", "Redes neurais", "Modelagem", "Dia 2"),
                ("NLTK", "Processamento de linguagem natural", "Tratamento/Modelagem", "Dia 2"),
            ],
            columns=["Biblioteca", "Para que serve", "Etapa do ciclo", "Quando"],
        ),
        hide_index=True,
    )

    st.header("2. Sintaxe e lógica")
    st.code(
        '''
# 1. Variáveis e tipos (tipagem dinâmica)
massa_kg = 1200.5          # float
modelo = "Cessna 172"      # str
em_voo = True              # bool

# 2. Estruturas de dados
velocidades_kt = [55, 65, 90, 110]                          # lista
aeronave = {"modelo": modelo, "massa_kg": massa_kg}         # dicionário

# 3. Controle de fluxo
if massa_kg > 1000:
    print("Aeronave de peso médio")
else:
    print("Aeronave leve")

for v in velocidades_kt:
    print(f"{v} kt = {v * 0.514444:.1f} m/s")

# 4. List comprehension
velocidades_ms = [v * 0.514444 for v in velocidades_kt]

# 5. Funções (com type hints e docstring)
def calcular_peso(massa_kg: float, g: float = 9.80665) -> float:
    """Retorna o peso em newtons."""
    return massa_kg * g
''',
        language="python",
    )

    st.subheader("Por que NumPy? Loop Python vs. operação vetorizada")
    if st.button("▶️ Rodar comparação (2 milhões de elementos)"):
        n = 2_000_000
        lista, arr = list(range(n)), np.arange(n)
        t0 = time.perf_counter()
        _ = [x * 2.0 for x in lista]
        t_lista = time.perf_counter() - t0
        t0 = time.perf_counter()
        _ = arr * 2.0
        t_np = time.perf_counter() - t0
        a, b, c = st.columns(3)
        a.metric("Lista + loop", f"{t_lista * 1000:.0f} ms")
        b.metric("NumPy vetorizado", f"{t_np * 1000:.1f} ms")
        c.metric("Ganho", f"{t_lista / max(t_np, 1e-9):.0f}×")

    st.subheader("🎯 Atividade")
    st.markdown("**Exercício 1.** Converta **120 kt** para km/h (1 kt = 1,852 km/h).")
    resp = st.number_input("Sua resposta (km/h)", value=0.0, key="ex1")
    if resp > 0:
        if abs(resp - 120 * 1.852) < 0.5:
            st.success("Correto! ✅")
        else:
            st.error("Ainda não. Reveja o fator de conversão.")
    st.markdown("**Exercício 2.** Escreva `energia_cinetica(massa_kg, v_ms)` e teste com 1200 kg a 60 m/s.")
    with st.expander("Ver solução"):
        st.code(
            "def energia_cinetica(massa_kg, v_ms):\n"
            "    return 0.5 * massa_kg * v_ms**2\n\n"
            "print(energia_cinetica(1200, 60))   # 2 160 000 J",
            language="python",
        )

with abas[2]:
    st.header("🔢 NumPy e Pandas: por que não só listas?")
    st.markdown("Listas Python são flexíveis demais para números: cada item é um objeto, os tipos podem se misturar e não há operação sobre o conjunto.")
    lista, arr = [1, 2], np.array([1, 2])
    st.code("lista = [1, 2]\narr = np.array([1, 2])\nprint(lista * 2)   # repete a lista\nprint(arr * 2)     # multiplica cada elemento", language="python")
    st.write("lista * 2 =", lista * 2, "| array * 2 =", (arr * 2).tolist())

    st.subheader("Criando arrays")
    st.code("np.array(1, 2, 3, 4)       # errado: faltam os colchetes\nb = np.array([1, 2, 3, 4])   # vetor\nc = np.array([[1.5, 2, 3], [4, 5, 6]])   # matriz: uma lista por linha", language="python")
    c = np.array([[1.5, 2, 3], [4, 5, 6]])
    st.write(c)

    st.subheader("Acesso e fatiamento (o fim do intervalo é exclusivo)")
    m = np.array([[0, 1, 2, 3], [10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33]])
    a = np.array([1, 2, 3, 4])
    st.code("a[0]; a[-1]        # primeiro e último\nc[1, 2]            # linha 1, coluna 2\na[1:3]             # índices 1 e 2\nm[:, 0]            # primeira coluna\nm[-1, :]           # última linha\nm[1:, :]           # sem a primeira linha", language="python")
    st.write("a[0], a[-1] =", int(a[0]), int(a[-1]), "| c[1, 2] =", c[1, 2], "| a[1:3] =", a[1:3].tolist())
    st.write("m[:, 0] =", m[:, 0].tolist(), "| m[-1, :] =", m[-1, :].tolist())

    st.subheader("Pandas: tabelas com nomes")
    df_ex = pd.DataFrame({"tempo_s": [0, 1, 2, 3], "altitude_m": [0, 120, 260, 410], "fase": ["Subida"] * 4})
    st.code('df = pd.DataFrame({"tempo_s": [0, 1, 2, 3], "altitude_m": [0, 120, 260, 410]})\ndf["altitude_m"]            # uma coluna\ndf[df["altitude_m"] > 100] # filtro', language="python")
    st.dataframe(df_ex[df_ex["altitude_m"] > 100], hide_index=True)

    st.subheader("Seu primeiro gráfico")
    st.code("x = np.arange(0, 10)\ny = x**2 - 2*x + 1\nplt.plot(x, y)\nplt.xlabel('x'); plt.ylabel('y'); plt.title('y = x² − 2x + 1')\nplt.show()   # no Streamlit: st.pyplot(fig)", language="python")
    x = np.arange(0, 10)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(x, x**2 - 2 * x + 1)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title("y = x² − 2x + 1")
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)

with abas[3]:
    st.header("✈️ Projeto: Aircraft Performance Calculator")
    st.markdown("Calculamos a **velocidade de estol** $V_s$ a partir do equilíbrio sustentação = peso:")
    st.latex(r"V_s = \sqrt{\frac{2\, n\, W}{\rho\, S\, C_{L_{max}}}} \qquad n = \frac{1}{\cos\phi}")
    st.caption("n = fator de carga; φ = ângulo de inclinação (bank) em curva coordenada nivelada.")

    col1, col2 = st.columns(2)
    with col1:
        massa = st.number_input("Massa (kg)", min_value=100.0, value=1100.0, step=50.0)
        area_asa = st.number_input("Área da asa S (m²)", min_value=1.0, value=16.2, step=0.5)
        cl_max = st.slider("C_Lmax (configuração limpa)", 0.8, 3.0, 1.5, 0.1)
    with col2:
        modo = st.radio("Densidade do ar", ["Atmosfera padrão (ISA) por altitude", "Valor manual"], horizontal=True)
        if modo.startswith("Atmosfera"):
            alt = st.slider("Altitude (m)", 0, 6000, 0, 100)
            rho = float(atmosfera_isa(alt)[2])
            st.caption(f"ρ(ISA, {alt} m) = {rho:.3f} kg/m³")
        else:
            rho = st.slider("ρ (kg/m³)", 0.5, 1.5, 1.225, 0.01)
        bank = st.slider("Inclinação φ (graus)", 0, 75, 0, 5)

    n_fator = 1 / np.cos(np.radians(bank))
    vs = float(v_estol(massa, area_asa, rho, cl_max))
    vs_b = float(v_estol(massa, area_asa, rho, cl_max, n_fator))

    m1, m2, m3 = st.columns(3)
    delta = f"+{(vs_b - vs) * 3.6:.1f} km/h (φ={bank}°)" if bank else None
    m1.metric("Vs (m/s)", f"{vs_b:.1f}", delta, delta_color="inverse")
    m2.metric("Vs (km/h)", f"{vs_b * 3.6:.1f}")
    m3.metric("Vs (kt)", f"{vs_b / KT_EM_MS:.1f}")
    st.caption(f"Fator de carga n = {n_fator:.2f}. Referência: Cessna 172 limpo ≈ 53 kt (Vs1).")

    st.subheader("Configurações de flape e variação com a altitude")
    configs = {"Limpo (flape 0°)": cl_max, "Decolagem (flape parcial)": cl_max + 0.3, "Pouso (flape total)": cl_max + 0.6}
    hs = np.linspace(0, 6000, 61)
    rhos = atmosfera_isa(hs)[2]
    sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots(figsize=(9, 3.6))
    tabela = []
    for nome, cl in configs.items():
        ax.plot(hs, v_estol(massa, area_asa, rhos, cl, n_fator) / KT_EM_MS, label=f"{nome} (CL={cl:.1f})")
        tabela.append({"Configuração": nome, "CLmax": round(cl, 1), "Vs a nível do mar (kt)": round(float(v_estol(massa, area_asa, 1.225, cl, n_fator)) / KT_EM_MS, 1)})
    ax.set_xlabel("Altitude (m)")
    ax.set_ylabel("Vs verdadeira (kt)")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.dataframe(pd.DataFrame(tabela), hide_index=True)
    st.caption("Obs.: aqui é velocidade **verdadeira** (TAS); a velocidade indicada (IAS) de estol quase não muda com a altitude.")

    with st.expander("Ver o código do cálculo"):
        st.code(
            "def v_estol(massa_kg, area_m2, rho, cl_max, fator_carga=1.0):\n"
            "    peso = massa_kg * 9.80665\n"
            "    return np.sqrt(2 * peso * fator_carga / (rho * area_m2 * cl_max))",
            language="python",
        )

    st.subheader("🎯 Atividade")
    st.markdown(
        "1. A que altitude ISA a Vs aumenta 10% em relação ao nível do mar? (use o gráfico)\n"
        "2. Quanto aumenta a Vs numa curva de 60° de inclinação? Confira o fator √n.\n"
        "3. **Extensão:** adicione um `st.number_input` de *massa de combustível* e mostre como a Vs cai ao longo do voo."
    )
    with st.expander("Respostas"):
        st.markdown("1. Aproximadamente 1,9 a 2,0 km (σ ≈ 0,83).  \n2. n = 2 → Vs × 1,41 (+41%).")

with abas[3]:
    st.subheader("🎯 Atividade: Aircraft Simulator")
    st.markdown(
        "Simule a velocidade de estol de uma aeronave leve:\n"
        "- Fórmula: $V_s = \\sqrt{2W / (\\rho S C_{L_{max}})}$, com $W = m \\cdot g$\n"
        "- Dados: $S = 16{,}2\\ m^2$, $C_{L_{max}} = 1{,}5$, massas de 800 a 1200 kg, $\\rho = 1{,}2\\ kg/m^3$\n"
        "- Meta: calcular a Vs para cada massa (sem `for`) e plotar Vs × massa."
    )
    st.code(
        "import numpy as np\nimport matplotlib.pyplot as plt\n\n"
        "S, CL_MAX, RHO, G = 16.2, 1.5, 1.2, 9.81\n"
        "massas = np.linspace(800, 1200, 5)     # kg\n"
        "# TODO 1: calcule vs (m/s) com np.sqrt, usando o array massas\n"
        "# TODO 2: converta para km/h (x 3.6)\n"
        "# TODO 3: plt.plot(massas, vs_kmh) com rótulos e unidades",
        language="python",
    )
    with st.expander("Ver gabarito"):
        st.code(
            "vs = np.sqrt(2 * massas * G / (RHO * S * CL_MAX))   # m/s\n"
            "vs_kmh = vs * 3.6\n"
            "plt.plot(massas, vs_kmh, marker='o')\n"
            "plt.xlabel('Massa (kg)'); plt.ylabel('Vs (km/h)'); plt.title('Velocidade de estol')\n"
            "plt.show()",
            language="python",
        )
        massas = np.linspace(800, 1200, 5)
        vs = np.sqrt(2 * massas * G0 / (1.2 * 16.2 * 1.5))
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(massas, vs * 3.6, marker="o")
        ax.set_xlabel("Massa (kg)"); ax.set_ylabel("Vs (km/h)"); ax.set_title("Velocidade de estol")
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)
        st.dataframe(pd.DataFrame({"massa_kg": massas, "Vs_m_s": vs.round(1), "Vs_km_h": (vs * 3.6).round(1), "Vs_kt": (vs / KT_EM_MS).round(1)}), hide_index=True)