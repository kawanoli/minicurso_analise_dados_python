"""Dia 1 – Fundamentos e o Ciclo do Dado (minicurso Python para Aeroespacial)

Execute com:  streamlit run dia1.py
"""
import sqlite3
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="Dia 1 – Fundamentos e Ciclo do Dado", page_icon="🐍", layout="wide")

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


@st.cache_data
def gerar_voo_bruto(seed=42):
    """Simula telemetria 'suja' de um voo de 10 min: ruído, NaN, outliers e linhas duplicadas."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, 601)
    alt = np.interp(t, [0, 200, 400, 600], [0, 3000, 3000, 0]) + rng.normal(0, 15, t.size)
    vel = np.interp(t, [0, 200, 400, 600], [60, 75, 75, 55]) + rng.normal(0, 1.5, t.size)
    temp = np.interp(t, [0, 200, 400, 600], [90, 105, 100, 85]) + rng.normal(0, 0.8, t.size)
    fase = np.select([t < 200, t < 400], ["Subida", "Cruzeiro"], "Descida")
    df = pd.DataFrame(
        {
            "tempo_s": t,
            "altitude_m": np.clip(alt, 0, None),
            "velocidade_ms": vel,
            "temp_motor_c": temp,
            "fase": fase,
        }
    )
    # --- sujeira proposital ---
    df.loc[rng.choice(t.size, 40, replace=False), "velocidade_ms"] = np.nan
    df.loc[rng.choice(t.size, 20, replace=False), "temp_motor_c"] = np.nan
    df.loc[[120, 310, 480], "altitude_m"] += [1500, -1200, 2000]  # spikes do sensor
    df = pd.concat([df, df.iloc[[50, 51, 52]]]).sort_values("tempo_s", kind="stable")
    return df.reset_index(drop=True)


def limpar(df, metodo_nan="interpolar", janela=7, limiar_m=300):
    """Pipeline de limpeza: duplicatas → outliers (mediana móvel) → NaN."""
    out = df.drop_duplicates().copy()
    n_dups = len(df) - len(out)

    mediana = out["altitude_m"].rolling(janela, center=True, min_periods=1).median()
    mascara_out = (out["altitude_m"] - mediana).abs() > limiar_m
    out.loc[mascara_out, "altitude_m"] = np.nan

    cols = ["altitude_m", "velocidade_ms", "temp_motor_c"]
    if metodo_nan == "interpolar":
        out[cols] = out[cols].interpolate(limit_direction="both")
    elif metodo_nan == "preencher com a média":
        out[cols] = out[cols].fillna(out[cols].mean())
    else:
        out = out.dropna(subset=cols)

    info = {"duplicatas": n_dups, "outliers": int(mascara_out.sum())}
    return out.reset_index(drop=True), info


# =====================================================================
# CABEÇALHO
# =====================================================================
st.title("🐍 Dia 1: Fundamentos e o Ciclo do Dado")
st.markdown(
    "Base de Python para engenharia e primeira volta completa no **ciclo da análise de dados**. "
    "Cada aba termina com uma **🎯 Atividade**."
)

abas = st.tabs(
    [
        "🧭 Visão geral",
        "🛠️ Setup e Python básico",
        "✈️ Aircraft Calculator",
        "📥 Coleta",
        "🧹 Tratamento e filtragem",
        "📊 Visualização",
    ]
)
df_bruto = gerar_voo_bruto()

# =====================================================================
# ABA 0 – VISÃO GERAL
# =====================================================================
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

# =====================================================================
# ABA 1 – SETUP E PYTHON BÁSICO
# =====================================================================
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

# =====================================================================
# ABA 2 – AIRCRAFT PERFORMANCE CALCULATOR
# =====================================================================
with abas[2]:
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

# =====================================================================
# ABA 3 – COLETA
# =====================================================================
with abas[3]:
    st.header("📥 Etapa 1: Coleta de dados")
    st.markdown("De onde vêm os dados? Estas são as fontes mais comuns e as ferramentas Python para cada uma.")
    st.dataframe(
        pd.DataFrame(
            [
                ("Arquivos locais", "pandas.read_csv / read_excel / read_json", "Logs de ensaio em voo exportados do sistema de aquisição", "Encoding, separador, vírgula decimal"),
                ("APIs REST", "requests, httpx", "OpenSky Network (posições de aeronaves), aviationweather.gov (METAR/TAF), APIs abertas da NASA", "Chaves de API, limite de requisições, termos de uso"),
                ("Raspagem web", "requests + BeautifulSoup, pandas.read_html, Playwright", "Tabelas de especificações de aeronaves em páginas públicas", "robots.txt, termos de uso, mudanças de layout"),
                ("Bancos de dados", "sqlite3, SQLAlchemy, pandas.read_sql", "Histórico de manutenção armazenado em SQL", "Consultas parametrizadas, permissões"),
                ("Sensores e telemetria ao vivo", "pyserial, pymavlink, paho-mqtt", "Telemetria de drones (ArduPilot/PX4), microcontroladores", "Taxa de amostragem, timestamps, perda de pacotes"),
                ("Datasets públicos", "pandas, kaggle, urllib", "NASA C-MAPSS (degradação de turbofans), dados abertos da ANAC e do CENIPA", "Licença, documentação, vieses"),
            ],
            columns=["Fonte", "Ferramentas Python", "Exemplo aeroespacial", "Cuidados"],
        ),
        hide_index=True,
    )

    st.subheader("Demo A: arquivo CSV")
    arquivo = st.file_uploader("Envie um CSV seu (ou use o dataset de telemetria simulado)", type="csv")
    if arquivo is not None:
        df_demo = pd.read_csv(arquivo)
        st.success(f"{len(df_demo)} linhas lidas.")
    else:
        df_demo = df_bruto
        st.caption("Usando a telemetria simulada de um voo de 10 minutos.")
    st.dataframe(df_demo.head(8))
    st.download_button("⬇️ Baixar telemetria simulada (CSV)", df_bruto.to_csv(index=False), "telemetria_bruta.csv", "text/csv")

    st.subheader("Demo B: banco de dados SQL (SQLite em memória)")
    query = st.text_area(
        "Consulta SQL",
        "SELECT fase,\n       COUNT(*) AS amostras,\n       ROUND(AVG(velocidade_ms), 1) AS vel_media,\n       MAX(altitude_m) AS alt_max\nFROM telemetria\nGROUP BY fase",
        height=140,
    )
    con = sqlite3.connect(":memory:")
    df_bruto.to_sql("telemetria", con, index=False)
    try:
        st.dataframe(pd.read_sql_query(query, con), hide_index=True)
    except Exception as e:  # noqa: BLE001
        st.error(f"Erro na consulta: {e}")
    finally:
        con.close()

    st.subheader("Demo C: API e raspagem (código de referência)")
    st.code(
        '''
import requests, pandas as pd

# API: aeronaves sobre o Nordeste brasileiro (OpenSky Network, bounding box)
url = "https://opensky-network.org/api/states/all"
params = {"lamin": -9.0, "lomin": -38.0, "lamax": -4.0, "lomax": -34.0}
resp = requests.get(url, params=params, timeout=15)
resp.raise_for_status()
estados = resp.json()["states"]            # lista de listas

# Raspagem: tabelas HTML direto para DataFrame
tabelas = pd.read_html("https://exemplo.org/especificacoes-aeronaves")
df_specs = tabelas[0]
''',
        language="python",
    )
    st.warning("⚖️ Antes de raspar: leia o `robots.txt`, os termos de uso e prefira API oficial quando existir.")

    st.subheader("🎯 Atividade")
    st.markdown(
        "1. Escreva uma consulta SQL que retorne só a fase **Cruzeiro** com temperatura média do motor.\n"
        "2. Escolha uma fonte da tabela e descreva: quais colunas você esperaria? Que problemas de qualidade?\n"
        "3. **Extensão (casa):** baixe o C-MAPSS (FD001) e carregue com `pd.read_csv(..., sep=r'\\s+', header=None)`."
    )
    with st.expander("Dica para o item 1"):
        st.code("SELECT AVG(temp_motor_c) FROM telemetria WHERE fase = 'Cruzeiro'", language="sql")

# =====================================================================
# ABA 4 – TRATAMENTO E FILTRAGEM
# =====================================================================
with abas[4]:
    st.header("🧹 Etapa 2: Tratamento e filtragem")
    st.markdown("Dados reais chegam com **NaN**, **outliers** e **duplicatas**. Veja o diagnóstico do dado bruto:")

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Linhas", len(df_bruto))
    d2.metric("Valores NaN", int(df_bruto.isna().sum().sum()))
    d3.metric("Linhas duplicadas", int(df_bruto.duplicated().sum()))
    d4.metric("Altitude máx. (m)", f"{df_bruto['altitude_m'].max():.0f}", help="Voo real só chega a ~3000 m: há spikes!")
    st.code("df.info()\ndf.isna().sum()\ndf.duplicated().sum()\ndf.describe()", language="python")

    st.subheader("Limpeza interativa")
    c1, c2, c3 = st.columns(3)
    metodo = c1.radio("Valores faltantes", ["interpolar", "preencher com a média", "descartar"])
    limiar = c2.slider("Limiar de outlier (m)", 100, 1000, 300, 50)
    janela = c3.select_slider("Janela da mediana móvel", options=[3, 5, 7, 9, 11], value=7)
    df_limpo, info = limpar(df_bruto, metodo, janela, limiar)
    st.success(f"Removidas {info['duplicatas']} duplicatas · {info['outliers']} outliers detectados · {len(df_limpo)} linhas finais")

    sns.set_theme(style="darkgrid")
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    for ax, col, rotulo in zip(axes, ["altitude_m", "velocidade_ms"], ["Altitude (m)", "Velocidade (m/s)"]):
        ax.plot(df_bruto["tempo_s"], df_bruto[col], color="lightgray", label="bruto")
        ax.plot(df_limpo["tempo_s"], df_limpo[col], color="tab:blue", linewidth=1.5, label="limpo")
        ax.set_ylabel(rotulo)
        ax.legend(loc="upper right")
    axes[1].set_xlabel("Tempo (s)")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Experimente 'preencher com a média': o que acontece nos pontos que eram outliers? Por quê?")

    st.code(
        '''
df = df.drop_duplicates()
mediana = df["altitude_m"].rolling(7, center=True, min_periods=1).median()
outlier = (df["altitude_m"] - mediana).abs() > 300
df.loc[outlier, "altitude_m"] = np.nan
df[["altitude_m", "velocidade_ms", "temp_motor_c"]] = (
    df[["altitude_m", "velocidade_ms", "temp_motor_c"]].interpolate()
)''',
        language="python",
    )

    st.subheader("Filtragem")
    f1, f2 = st.columns(2)
    fases = f1.multiselect("Fases do voo", ["Subida", "Cruzeiro", "Descida"], default=["Cruzeiro"])
    t0, t1 = f2.slider("Janela de tempo (s)", 0, 600, (0, 600))
    mascara = df_limpo["fase"].isin(fases) & df_limpo["tempo_s"].between(t0, t1)
    df_f = df_limpo[mascara]
    st.write(f"**{len(df_f)}** amostras selecionadas")
    st.dataframe(df_f.describe().round(1))
    st.code(
        'mascara = df["fase"].isin(fases) & df["tempo_s"].between(t0, t1)\n'
        'df_filtrado = df[mascara]\n\n'
        '# equivalente com query():\n'
        'df.query("fase in @fases and @t0 <= tempo_s <= @t1")',
        language="python",
    )

    st.subheader("🎯 Atividade")
    st.markdown(
        "1. Qual método de NaN gera o gráfico mais fiel? Justifique.\n"
        "2. Qual a velocidade média na **Subida**? E a temperatura máxima na **Descida**?\n"
        "3. Crie uma coluna `vel_kt` e filtre amostras acima de 140 kt (ela existe?)."
    )
    with st.expander("Respostas"):
        st.code(
            'df_limpo.groupby("fase")["velocidade_ms"].mean()\n'
            'df_limpo.loc[df_limpo["fase"] == "Descida", "temp_motor_c"].max()\n'
            'df_limpo["vel_kt"] = df_limpo["velocidade_ms"] / 0.514444\n'
            '(df_limpo["vel_kt"] > 140).any()   # False',
            language="python",
        )

# =====================================================================
# ABA 5 – VISUALIZAÇÃO
# =====================================================================
with abas[5]:
    st.header("📊 Etapa 4: Visualização (Matplotlib + Seaborn)")
    st.markdown("Gráficos estáticos são ideais para relatórios. No Dia 2 veremos interatividade com Plotly.")
    df_v, _ = limpar(df_bruto)
    tipo = st.selectbox(
        "Tipo de gráfico",
        ["Série temporal (eixo duplo)", "Dispersão altitude × velocidade", "Boxplot por fase", "Histograma + KDE", "Mapa de correlação"],
    )
    sns.set_theme(style="whitegrid")  # ANTES de criar a figura
    fig, ax = plt.subplots(figsize=(10, 4))
    if tipo.startswith("Série"):
        ax.plot(df_v["tempo_s"], df_v["altitude_m"], color="tab:blue", linewidth=2)
        ax.set_xlabel("Tempo (s)")
        ax.set_ylabel("Altitude (m)", color="tab:blue")
        ax2 = ax.twinx()
        ax2.plot(df_v["tempo_s"], df_v["velocidade_ms"], color="tab:red", linestyle="--")
        ax2.set_ylabel("Velocidade (m/s)", color="tab:red")
        ax2.grid(False)
    elif tipo.startswith("Dispersão"):
        sns.scatterplot(data=df_v, x="velocidade_ms", y="altitude_m", hue="fase", ax=ax)
    elif tipo.startswith("Boxplot"):
        sns.boxplot(data=df_v, x="fase", y="temp_motor_c", order=["Subida", "Cruzeiro", "Descida"], ax=ax)
    elif tipo.startswith("Histograma"):
        sns.histplot(df_v["velocidade_ms"], kde=True, bins=30, ax=ax)
    else:
        sns.heatmap(df_v[["altitude_m", "velocidade_ms", "temp_motor_c"]].corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Boas práticas")
    st.markdown(
        "- Sempre rotule eixos **com unidades**\n"
        "- Cor com significado (não decorativa); use eixo duplo com parcimônia\n"
        "- Configure o tema (`sns.set_theme`) **antes** de criar a figura\n"
        "- Feche figuras (`plt.close(fig)`) em apps para não vazar memória"
    )

    st.subheader("🎯 Mini-projeto do Dia 1")
    st.markdown(
        "Com o CSV da aba *Coleta*, entregue um notebook ou script que:\n"
        "1. Carrega e diagnostica o dado (NaN, duplicatas, outliers)\n"
        "2. Limpa e documenta cada decisão\n"
        "3. Filtra a fase de **Cruzeiro**\n"
        "4. Produz uma figura com 2 painéis, com unidades e título"
    )