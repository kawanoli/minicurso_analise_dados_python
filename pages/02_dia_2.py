"""Dia 2 – Aeroespacial, Modelagem e Inteligência Artificial (minicurso Python para Aeroespacial)

Execute com:  streamlit run dia2.py
"""
import importlib.util
import io
import os
import tempfile

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.formula.api as smf
import streamlit as st
from nltk import FreqDist
from nltk.tokenize import RegexpTokenizer
from plotly.subplots import make_subplots
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Dia 2 – Aeroespacial e IA", page_icon="🚀", layout="wide")

G0 = 9.80665
FEATURES = ["peso_kg", "altitude_m", "velocidade_kmh"]
ALVO = "consumo_kgh"


# =====================================================================
# FUNÇÕES AUXILIARES
# =====================================================================
def atmosfera_isa(h):
    h = np.asarray(h, dtype=float)
    T = np.where(h <= 11000, 288.15 - 0.0065 * h, 216.65)
    p = np.where(
        h <= 11000,
        101325 * (T / 288.15) ** 5.2559,
        22632.06 * np.exp(-G0 * (h - 11000) / (287.058 * 216.65)),
    )
    return T, p, p / (287.058 * T)


@st.cache_data
def gerar_consumo(n=400, seed=42):
    """Dataset sintético: consumo depende de peso, altitude e de uma INTERAÇÃO peso×altitude.
    A velocidade é irrelevante de propósito (para discutir p-valores)."""
    rng = np.random.default_rng(seed)
    peso = rng.uniform(1000, 5000, n)
    alt = rng.uniform(2000, 10000, n)
    vel = rng.uniform(200, 300, n)
    consumo = 0.05 * peso + 0.01 * alt + 5e-6 * peso * alt + rng.normal(0, 15, n)
    return pd.DataFrame({"peso_kg": peso, "altitude_m": alt, "velocidade_kmh": vel, ALVO: consumo})


@st.cache_data
def gerar_telemetria(n=100_000):
    rng = np.random.default_rng(0)
    t = np.arange(n) / 100.0  # 100 Hz
    return pd.DataFrame(
        {
            "t_s": t,
            "temp_motor_esq": 90 + 5 * np.sin(t / 50) + rng.normal(0, 0.5, n),
            "temp_motor_dir": 91 + 5 * np.sin(t / 50 + 0.2) + rng.normal(0, 0.5, n),
            "rpm": 2300 + 50 * np.sin(t / 20) + rng.normal(0, 8, n),
            "acc_z_g": 1 + rng.normal(0, 0.05, n),
        }
    )


@st.cache_data
def comparar_formatos():
    """Grava a mesma telemetria em vários formatos e mede o tamanho."""
    df = gerar_telemetria()
    linhas = [{"Formato": "CSV", "Tamanho (MB)": len(df.to_csv(index=False).encode()) / 1e6}]
    try:
        buf = io.BytesIO()
        df.to_parquet(buf)
        linhas.append({"Formato": "Parquet", "Tamanho (MB)": buf.getbuffer().nbytes / 1e6})
    except Exception:  # noqa: BLE001  (pyarrow ausente)
        pass
    estrutura, trecho = [], []
    if importlib.util.find_spec("h5py"):
        import h5py

        with tempfile.TemporaryDirectory() as d:
            caminho = os.path.join(d, "telemetria.h5")
            with h5py.File(caminho, "w") as f:
                f.attrs["aeronave"] = "PP-XYZ (fictícia)"
                f.attrs["taxa_hz"] = 100
                for lado, col in [("esquerdo", "temp_motor_esq"), ("direito", "temp_motor_dir")]:
                    ds = f.create_group(f"motores/{lado}").create_dataset(
                        "temperatura", data=df[col].values.astype("float32"), compression="gzip"
                    )
                    ds.attrs["unidade"] = "°C"
                f.create_dataset("dinamica/rpm", data=df["rpm"].values.astype("float32"), compression="gzip")
                f.create_dataset("dinamica/acc_z_g", data=df["acc_z_g"].values.astype("float32"), compression="gzip")
            linhas.append({"Formato": "HDF5 (float32 + gzip)", "Tamanho (MB)": os.path.getsize(caminho) / 1e6})
            with h5py.File(caminho, "r") as f:
                def visita(nome, obj):
                    if isinstance(obj, h5py.Dataset):
                        estrutura.append(f"📄 /{nome}   shape={obj.shape}  dtype={obj.dtype}  attrs={dict(obj.attrs)}")
                    else:
                        estrutura.append(f"📁 /{nome}")

                f.visititems(visita)
                trecho = f["motores/esquerdo/temperatura"][:5].tolist()
    return pd.DataFrame(linhas), estrutura, trecho


@st.cache_resource
def avaliar_modelos(test_size, n_arvores):
    df = gerar_consumo()
    X, y = df[FEATURES], df[ALVO]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=42)
    modelos = {
        "Regressão Linear": LinearRegression(),
        "Ridge (padronizado)": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Random Forest": RandomForestRegressor(n_estimators=n_arvores, random_state=42),
    }
    linhas, preds = [], {}
    for nome, m in modelos.items():
        m.fit(X_tr, y_tr)
        p = m.predict(X_te)
        linhas.append(
            {
                "Modelo": nome,
                "MAE (kg/h)": mean_absolute_error(y_te, p),
                "RMSE (kg/h)": float(np.sqrt(mean_squared_error(y_te, p))),
                "R² treino": m.score(X_tr, y_tr),
                "R² validação cruzada": cross_val_score(m, X_tr, y_tr, cv=5, scoring="r2").mean(),
                "R² teste": r2_score(y_te, p),
            }
        )
        preds[nome] = p
    return pd.DataFrame(linhas).set_index("Modelo"), preds, y_te.values, modelos


# ---------- NLP ----------
LOGS = {
    "Motor": [
        "Vibração excessiva no motor esquerdo durante a decolagem",
        "Temperatura de óleo do motor acima do limite em cruzeiro",
        "Substituição do filtro de combustível do motor direito",
        "Perda de potência do motor na subida, verificar injetores",
        "Fumaça no escapamento após partida do motor",
        "Pressão de óleo oscilando no motor durante o táxi",
        "Inspeção boroscópica nas pás da turbina revelou desgaste",
        "Ruído anormal no motor durante a redução de potência",
    ],
    "Hidráulico": [
        "Vazamento de fluido hidráulico no trem de pouso principal",
        "Pressão do sistema hidráulico abaixo do normal após o pouso",
        "Substituição da bomba hidráulica do sistema B",
        "Atuador do flape com resposta lenta, suspeita de ar no circuito hidráulico",
        "Reservatório hidráulico com nível baixo, reabastecido",
        "Mangueira hidráulica do freio com sinais de desgaste",
        "Trem de pouso demorou a retrair, verificar pressão hidráulica",
        "Vazamento de fluido no atuador do profundor",
    ],
    "Elétrico": [
        "Falha intermitente no gerador elétrico durante o voo",
        "Bateria descarregada após a parada, substituída",
        "Pane no painel de instrumentos, queda de tensão no barramento",
        "Disjuntor do transponder desarmando em voo",
        "Curto-circuito detectado no chicote elétrico da luz de navegação",
        "Alternador não carrega, regulador de tensão substituído",
        "Luz de aviso do painel piscando, mau contato no conector",
        "Rádio sem energia, fusível queimado",
    ],
}
STOP_FALLBACK = {"de", "do", "da", "dos", "das", "no", "na", "nos", "nas", "em", "para", "com", "por", "um", "uma", "o", "a", "os", "as", "e", "ou", "que", "se", "ao", "após", "durante"}
TOK = RegexpTokenizer(r"[a-zA-ZÀ-ÿ]+")


@st.cache_resource
def carregar_stopwords():
    try:
        import nltk
        from nltk.corpus import stopwords

        try:
            return set(stopwords.words("portuguese")), "corpus NLTK"
        except LookupError:
            nltk.download("stopwords", quiet=True)
            return set(stopwords.words("portuguese")), "corpus NLTK (baixado agora)"
    except Exception:  # noqa: BLE001
        return STOP_FALLBACK, "lista reduzida local (corpus NLTK indisponível)"


def preprocessar(texto):
    stops, _ = carregar_stopwords()
    return [t for t in TOK.tokenize(texto.lower()) if t not in stops and len(t) > 2]


@st.cache_resource
def treinar_classificador():
    textos = [t for lst in LOGS.values() for t in lst]
    rotulos = [k for k, lst in LOGS.items() for _ in lst]
    pipe = make_pipeline(
        TfidfVectorizer(tokenizer=preprocessar, lowercase=False, token_pattern=None),
        LogisticRegression(max_iter=1000),
    )
    acc = cross_val_score(pipe, textos, rotulos, cv=4, scoring="accuracy").mean()
    pipe.fit(textos, rotulos)
    return pipe, acc


# ---------- Mecânica orbital ----------
MU, RE = 398600.4418, 6378.137  # km³/s², km


def hohmann(h1, h2):
    r1, r2 = RE + h1, RE + h2
    at = (r1 + r2) / 2
    v1, v2 = np.sqrt(MU / r1), np.sqrt(MU / r2)
    vp, va = np.sqrt(MU * (2 / r1 - 1 / at)), np.sqrt(MU * (2 / r2 - 1 / at))
    return {"r1": r1, "r2": r2, "a": at, "dv1": vp - v1, "dv2": v2 - va, "tof_h": np.pi * np.sqrt(at**3 / MU) / 3600}


# =====================================================================
# CABEÇALHO
# =====================================================================
st.title("🚀 Dia 2: Aeroespacial e Inteligência Artificial")
st.markdown(
    "Hoje fechamos o ciclo: **I/O de dados pesados**, bibliotecas do setor, **modelagem** (SciPy, statsmodels, "
    "scikit-learn, TensorFlow), **NLP** e visualização interativa."
)
abas = st.tabs(
    [
        "🗂️ Data I/O e formatos",
        "🛰️ Bibliotecas aeroespaciais",
        "📈 Modelagem e viz interativa",
        "📐 statsmodels",
        "🤖 scikit-learn",
        "🧠 TensorFlow",
        "💬 NLTK",
        "🏁 Projeto integrador",
    ]
)
df_consumo = gerar_consumo()

# =====================================================================
# ABA 0 – DATA I/O
# =====================================================================
with abas[0]:
    st.header("Data I/O: escolhendo o formato certo")
    st.dataframe(
        pd.DataFrame(
            [
                ("CSV / TSV", "Texto simples", "Universal, legível", "Grande, lento, sem tipos nem metadados"),
                ("JSON", "Texto hierárquico", "APIs, configs", "Verboso; ruim para tabelas grandes"),
                ("Parquet", "Colunar binário", "Analytics com Pandas/Spark", "Não é legível por humanos"),
                ("HDF5", "Hierárquico binário", "Telemetria em alta frequência, múltiplos canais", "Escrita concorrente é delicada"),
                ("NetCDF", "Hierárquico (baseado em HDF5)", "Dados científicos em grade (clima, satélite)", "Curva de aprendizado"),
                ("SQL", "Relacional", "Dados transacionais e consultas", "Não é ideal para séries muito densas"),
            ],
            columns=["Formato", "Tipo", "Quando usar", "Limitações"],
        ),
        hide_index=True,
    )

    st.subheader("Experimento: 100 mil amostras a 100 Hz (≈ 17 min de voo, 5 canais)")
    tabela, estrutura, trecho = comparar_formatos()
    c1, c2 = st.columns([1, 1])
    with c1:
        st.dataframe(tabela.round(2), hide_index=True)
        st.bar_chart(tabela.set_index("Formato"))
    with c2:
        if estrutura:
            st.markdown("**Estrutura do arquivo HDF5** (um “sistema de arquivos dentro de um arquivo”):")
            st.code("\n".join(estrutura))
            st.caption(f"Leitura parcial (só 5 amostras, sem carregar o resto): {np.round(trecho, 2).tolist()}")
        else:
            st.warning("Instale `h5py` para ver a demonstração de HDF5.")

    st.code(
        '''
import h5py

with h5py.File("telemetria_ensaio_01.h5", "r") as f:
    temp_esq = f["motores/esquerdo/temperatura"][:]          # lê tudo
    trecho   = f["motores/esquerdo/temperatura"][1000:2000]  # lê só um trecho
    unidade  = f["motores/esquerdo/temperatura"].attrs["unidade"]
''',
        language="python",
    )

    st.subheader("🎯 Atividade")
    st.markdown(
        "1. Por que o HDF5 ficou menor que o CSV? Cite **duas** razões (dica: tipo numérico e compressão).\n"
        "2. Grave o DataFrame de telemetria em Parquet e HDF5 e compare o tempo de **leitura** com `%timeit`.\n"
        "3. Que metadados você guardaria em `attrs` para um ensaio em voo?"
    )

# =====================================================================
# ABA 1 – BIBLIOTECAS AEROESPACIAIS
# =====================================================================
with abas[1]:
    st.header("O ecossistema Python aeroespacial")
    st.dataframe(
        pd.DataFrame(
            [
                ("astropy", "Unidades, tempo, coordenadas astronômicas", "Evita erros de unidade"),
                ("poliastro → hapsira", "Mecânica orbital interativa", "Atenção: o poliastro foi descontinuado; hapsira é um fork mantido pela comunidade"),
                ("skyfield / sgp4", "Posição de satélites a partir de TLE", "Rastreamento de passagens"),
                ("AeroSandbox", "Projeto e otimização de aeronaves", "Diferenciação automática"),
                ("OpenAP", "Modelos de desempenho de aeronaves", "Dados abertos da TU Delft"),
                ("ambiance", "Atmosfera ISA", "Alternativa à função própria"),
                ("pymavlink / pyulog", "Telemetria e logs de drones (MAVLink, PX4)", "Ponte com o curso de firmware"),
                ("traffic", "Análise de dados de tráfego aéreo", "Trajetórias ADS-B"),
            ],
            columns=["Biblioteca", "Para que serve", "Observação"],
        ),
        hide_index=True,
    )
    st.caption("Material-base do curso: https://pythonforaerospace.com")

    st.subheader("Unidades com astropy")
    if importlib.util.find_spec("astropy"):
        from astropy import units as u

        altura = 35000 * u.imperial.ft
        st.write(f"35 000 ft = **{altura.to(u.m):.0f}** = **{altura.to(u.km):.2f}**")
        st.code(
            "from astropy import units as u\nh = 35000 * u.imperial.ft\nh.to(u.m)\n"
            "# 10*u.m + 5*u.s  → UnitConversionError: unidades incompatíveis!",
            language="python",
        )
    else:
        st.info("Instale `astropy` para ver esta demonstração.")

    st.subheader("Atmosfera ISA")
    h = np.linspace(0, 20000, 200)
    T, p, rho = atmosfera_isa(h)
    fig = make_subplots(rows=1, cols=3, shared_yaxes=True, subplot_titles=("Temperatura (°C)", "Pressão (kPa)", "Densidade (kg/m³)"))
    for i, (x, nome) in enumerate([(T - 273.15, "T"), (p / 1000, "p"), (rho, "ρ")], start=1):
        fig.add_trace(go.Scatter(x=x, y=h, name=nome, showlegend=False), row=1, col=i)
    fig.update_yaxes(title_text="Altitude (m)", row=1, col=1)
    fig.update_layout(height=380, margin=dict(t=40, b=10))
    st.plotly_chart(fig)

    st.subheader("Transferência de Hohmann (sem biblioteca externa)")
    c1, c2 = st.columns(2)
    h1 = c1.slider("Altitude da órbita inicial (km)", 200, 2000, 400, 50)
    h2 = c2.slider("Altitude da órbita final (km)", 2500, 40000, 35786, 100)
    r = hohmann(h1, h2)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Δv₁ (km/s)", f"{r['dv1']:.2f}")
    m2.metric("Δv₂ (km/s)", f"{r['dv2']:.2f}")
    m3.metric("Δv total (km/s)", f"{r['dv1'] + r['dv2']:.2f}")
    m4.metric("Tempo de transferência (h)", f"{r['tof_h']:.1f}")
    th = np.linspace(0, 2 * np.pi, 400)
    e = (r["r2"] - r["r1"]) / (r["r2"] + r["r1"])
    ang = np.linspace(0, np.pi, 200)
    rt = r["a"] * (1 - e**2) / (1 + e * np.cos(ang))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=RE * np.cos(th), y=RE * np.sin(th), fill="toself", name="Terra", line=dict(color="royalblue")))
    fig.add_trace(go.Scatter(x=r["r1"] * np.cos(th), y=r["r1"] * np.sin(th), name="Órbita inicial"))
    fig.add_trace(go.Scatter(x=r["r2"] * np.cos(th), y=r["r2"] * np.sin(th), name="Órbita final"))
    fig.add_trace(go.Scatter(x=rt * np.cos(ang), y=rt * np.sin(ang), name="Elipse de transferência", line=dict(dash="dash")))
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_layout(height=500, xaxis_title="x (km)", yaxis_title="y (km)")
    st.plotly_chart(fig)
    st.caption("LEO 400 km → GEO: Δv total ≈ 3,9 km/s (confira com os valores padrão).")

    with st.expander("Mesmo cálculo com biblioteca (hapsira/poliastro) — código de referência"):
        st.code(
            '''
from astropy import units as u
from poliastro.bodies import Earth        # no hapsira: from hapsira.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.maneuver import Maneuver

leo = Orbit.circular(Earth, alt=400 * u.km)
hoh = Maneuver.hohmann(leo, Earth.R + 35786 * u.km)
print(hoh.get_total_cost())
''',
            language="python",
        )

    st.subheader("🎯 Atividade")
    st.markdown(
        "1. Qual o Δv total para ir de 400 km até 20 200 km (órbita de GPS)?\n"
        "2. Por que a densidade quase não muda acima de ~15 km na curva ISA acima? Compare com a pressão.\n"
        "3. **Extensão:** implemente a velocidade do som `a = √(γ R T)` e plote o Mach de uma aeronave a 250 m/s vs. altitude."
    )

# =====================================================================
# ABA 2 – MODELAGEM MATEMÁTICA + VIZ INTERATIVA
# =====================================================================
with abas[2]:
    st.header("Modelagem matemática com dados: a polar de arrasto")
    st.markdown(
        "Em ensaios em túnel de vento medimos $C_L$ e $C_D$. O modelo parabólico clássico é:"
    )
    st.latex(r"C_D = C_{D0} + k\, C_L^2 \qquad \left(\frac{L}{D}\right)_{max} = \frac{1}{2\sqrt{C_{D0}\,k}}")
    c1, c2, c3 = st.columns(3)
    n_pts = c1.slider("Nº de pontos ensaiados", 8, 200, 40)
    ruido = c2.slider("Ruído de medição em CD", 0.0005, 0.01, 0.002, 0.0005, format="%.4f")
    semente = c3.number_input("Semente aleatória", 0, 999, 1)

    CD0_V, K_V = 0.030, 0.055
    rng = np.random.default_rng(int(semente))
    cl = rng.uniform(0.0, 1.4, n_pts)
    cd = CD0_V + K_V * cl**2 + rng.normal(0, ruido, n_pts)

    popt, pcov = curve_fit(lambda x, cd0, k: cd0 + k * x**2, cl, cd, p0=[0.02, 0.05])
    erro = np.sqrt(np.diag(pcov))
    cd0_e, k_e = popt
    pred = cd0_e + k_e * cl**2
    r2 = 1 - np.sum((cd - pred) ** 2) / np.sum((cd - cd.mean()) ** 2)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("CD0 estimado", f"{cd0_e:.4f}", f"± {erro[0]:.4f} (real {CD0_V})", delta_color="off")
    m2.metric("k estimado", f"{k_e:.4f}", f"± {erro[1]:.4f} (real {K_V})", delta_color="off")
    m3.metric("(L/D)max", f"{1 / (2 * np.sqrt(cd0_e * k_e)):.1f}", f"real {1 / (2 * np.sqrt(CD0_V * K_V)):.1f}", delta_color="off")
    m4.metric("R²", f"{r2:.3f}")

    xs = np.linspace(0, 1.4, 100)
    g1, g2 = st.columns(2)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cd, y=cl, mode="markers", name="Medições"))
    fig.add_trace(go.Scatter(x=cd0_e + k_e * xs**2, y=xs, name="Ajuste", line=dict(color="red")))
    fig.update_layout(title="Polar de arrasto", xaxis_title="CD", yaxis_title="CL", height=400)
    g1.plotly_chart(fig)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cl[cl > 0.05], y=cl[cl > 0.05] / cd[cl > 0.05], mode="markers", name="Medições"))
    fig.add_trace(go.Scatter(x=xs[1:], y=xs[1:] / (cd0_e + k_e * xs[1:] ** 2), name="Modelo", line=dict(color="red")))
    fig.update_layout(title="Eficiência aerodinâmica L/D", xaxis_title="CL", yaxis_title="L/D", height=400)
    g2.plotly_chart(fig)
    st.caption("Reduza o número de pontos e aumente o ruído: as barras de erro (±) crescem. Esse é o custo de poucos dados.")

    st.code(
        '''
from scipy.optimize import curve_fit

def polar(cl, cd0, k):
    return cd0 + k * cl**2

popt, pcov = curve_fit(polar, cl, cd, p0=[0.02, 0.05])
cd0, k = popt
incerteza = np.sqrt(np.diag(pcov))     # 1 desvio-padrão de cada parâmetro
''',
        language="python",
    )

    with st.expander("🌀 Visualização 3D interativa (trajetória)"):
        t = np.linspace(0, 20, 150)
        df_traj = pd.DataFrame({"X (Leste)": np.cos(t) * t, "Y (Norte)": np.sin(t) * t, "Z (Altitude)": t * 10, "Tempo": t})
        fig = px.line_3d(df_traj, x="X (Leste)", y="Y (Norte)", z="Z (Altitude)", color="Tempo", title="Subida em espiral — rotacione com o mouse")
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))
        st.plotly_chart(fig)

    st.subheader("🎯 Atividade")
    st.markdown(
        "1. Com ruído 0,01 e 10 pontos, o intervalo de CD0 ainda contém o valor real?\n"
        "2. Em que CL ocorre (L/D)max? Mostre que é $C_L^*=\\sqrt{C_{D0}/k}$.\n"
        "3. **Extensão:** ajuste um modelo com termo linear ($C_D = C_{D0} + k_1 C_L + k_2 C_L^2$). O termo extra melhora algo?"
    )

# =====================================================================
# ABA 3 – STATSMODELS
# =====================================================================
with abas[3]:
    st.header("statsmodels: modelar para **entender** (inferência)")
    st.markdown(
        "Enquanto o scikit-learn foca em **prever**, o statsmodels responde: *esta variável realmente importa? Com que confiança?*"
    )
    usar_inter = st.checkbox("Incluir interação peso × altitude", value=False)
    formula = "consumo_kgh ~ peso_kg + altitude_m + velocidade_kmh" + (" + peso_kg:altitude_m" if usar_inter else "")
    st.code(f'modelo = smf.ols("{formula}", data=df).fit()\nprint(modelo.summary())', language="python")
    res = smf.ols(formula, data=df_consumo).fit()

    a, b, c = st.columns(3)
    a.metric("R²", f"{res.rsquared:.3f}")
    b.metric("R² ajustado", f"{res.rsquared_adj:.3f}")
    c.metric("AIC (menor é melhor)", f"{res.aic:.0f}")

    ic = res.conf_int()
    tabela = pd.DataFrame({"coeficiente": res.params, "IC 2,5%": ic[0], "IC 97,5%": ic[1], "p-valor": res.pvalues})
    st.dataframe(tabela.style.format("{:.4g}"))
    st.caption("`velocidade_kmh` tem p-valor alto: não há evidência de efeito (e, de fato, o dado foi gerado sem ele).")

    with st.expander("Ver o relatório completo (summary)"):
        st.code(res.summary().as_text())

    st.subheader("Diagnóstico dos resíduos")
    d1, d2 = st.columns(2)
    fig = px.scatter(x=res.fittedvalues, y=res.resid, labels={"x": "Valor ajustado (kg/h)", "y": "Resíduo (kg/h)"}, title="Resíduos × ajustado")
    fig.add_hline(y=0, line_dash="dash")
    d1.plotly_chart(fig)
    (osm, osr), (slope, intercept, _) = stats.probplot(res.resid, dist="norm")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=osm, y=osr, mode="markers", name="Resíduos"))
    fig.add_trace(go.Scatter(x=osm, y=slope * np.array(osm) + intercept, name="Normal teórica"))
    fig.update_layout(title="Q-Q plot", xaxis_title="Quantis teóricos", yaxis_title="Quantis observados")
    d2.plotly_chart(fig)
    st.caption("Sem a interação, os resíduos mostram **curvatura**: sinal de que falta estrutura no modelo. Ative a interação e compare.")

    st.subheader("🎯 Atividade")
    st.markdown(
        "1. Compare R² ajustado e AIC com e sem interação. Qual modelo você escolheria?\n"
        "2. Interprete o coeficiente de `peso_kg` sem a interação: o que ele significa em kg/h por kg?\n"
        "3. Remova `velocidade_kmh` da fórmula. O que muda no R² ajustado?"
    )

# =====================================================================
# ABA 4 – SCIKIT-LEARN
# =====================================================================
with abas[4]:
    st.header("scikit-learn: modelar para **prever**")
    st.markdown("A regra de ouro: **avalie em dados que o modelo nunca viu** (conjunto de teste).")
    c1, c2 = st.columns(2)
    test_size = c1.slider("Fração de teste", 0.1, 0.5, 0.25, 0.05)
    n_arv = c2.slider("Árvores do Random Forest", 10, 300, 100, 10)
    resultados, preds, y_te, modelos = avaliar_modelos(test_size, n_arv)
    st.dataframe(resultados.style.format("{:.3f}"))
    st.caption(
        "Compare R² de treino vs. teste: uma grande diferença indica *overfitting*. "
        "A regressão linear não captura a interação peso×altitude; o Random Forest captura."
    )

    escolha = st.selectbox("Previsto vs. real", list(preds))
    fig = px.scatter(x=y_te, y=preds[escolha], labels={"x": "Consumo real (kg/h)", "y": "Consumo previsto (kg/h)"}, title=escolha)
    lim = [float(min(y_te.min(), preds[escolha].min())), float(max(y_te.max(), preds[escolha].max()))]
    fig.add_trace(go.Scatter(x=lim, y=lim, name="Previsão perfeita", line=dict(dash="dash", color="gray")))
    p1, p2 = st.columns(2)
    p1.plotly_chart(fig)
    imp = pd.Series(modelos["Random Forest"].feature_importances_, index=FEATURES).sort_values()
    p2.plotly_chart(px.bar(imp, orientation="h", labels={"value": "Importância", "index": ""}, title="Importância das variáveis (RF)"))

    st.subheader("Previsão interativa")
    q1, q2, q3 = st.columns(3)
    peso_in = q1.slider("Peso (kg)", 1000, 5000, 3000, 100)
    alt_in = q2.slider("Altitude (m)", 2000, 10000, 6000, 250)
    vel_in = q3.slider("Velocidade (km/h)", 200, 300, 250, 10)
    x_novo = pd.DataFrame([[peso_in, alt_in, vel_in]], columns=FEATURES)
    k1, k2 = st.columns(2)
    k1.metric("Regressão Linear", f"{modelos['Regressão Linear'].predict(x_novo)[0]:.1f} kg/h")
    k2.metric("Random Forest", f"{modelos['Random Forest'].predict(x_novo)[0]:.1f} kg/h")
    st.caption("Tente valores fora de 1000–5000 kg (só a Linear extrapola; árvores 'achatam' nos limites do treino).")

    st.code(
        '''
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)
modelo = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_tr, y_tr)
print("R² teste:", modelo.score(X_te, y_te))
''',
        language="python",
    )
    st.subheader("🎯 Atividade")
    st.markdown(
        "1. Aumente a fração de teste para 0,5. Como muda o R² do teste? E o de treino?\n"
        "2. Por que *não* devemos reportar o R² de treino como “precisão” do modelo?\n"
        "3. **Extensão:** padronize os dados **antes** do `train_test_split` e explique por que isso é vazamento de dados (*data leakage*)."
    )

# =====================================================================
# ABA 5 – TENSORFLOW
# =====================================================================
with abas[5]:
    st.header("TensorFlow / Keras: redes neurais")
    st.markdown(
        "Redes neurais brilham com dados **não estruturados**: imagens (CNN), séries temporais (LSTM) e texto. "
        "Para tabelas pequenas como a nossa, modelos clássicos costumam bastar — vamos treinar uma rede só para comparar."
    )
    tf_ok = importlib.util.find_spec("tensorflow") is not None
    if not tf_ok:
        st.warning("TensorFlow não está instalado neste ambiente (`pip install tensorflow`). O código abaixo é de referência.")
    c1, c2 = st.columns(2)
    epocas = c1.slider("Épocas", 5, 200, 60, 5)
    neuronios = c2.select_slider("Neurônios por camada", [8, 16, 32, 64], value=32)
    if st.button("🏋️ Treinar rede neural", disabled=not tf_ok):
        import tensorflow as tf

        tf.keras.utils.set_random_seed(42)
        X, y = df_consumo[FEATURES], df_consumo[ALVO]
        X_tr, X_te, y_tr, y_te_ = train_test_split(X, y, test_size=0.25, random_state=42)
        sx, sy = StandardScaler().fit(X_tr), StandardScaler().fit(y_tr.values.reshape(-1, 1))
        rede = tf.keras.Sequential(
            [
                tf.keras.Input(shape=(3,)),
                tf.keras.layers.Dense(neuronios, activation="relu"),
                tf.keras.layers.Dense(neuronios, activation="relu"),
                tf.keras.layers.Dense(1),
            ]
        )
        rede.compile(optimizer="adam", loss="mse")
        with st.spinner("Treinando..."):
            hist = rede.fit(sx.transform(X_tr), sy.transform(y_tr.values.reshape(-1, 1)), validation_split=0.2, epochs=epocas, batch_size=32, verbose=0)
        pred = sy.inverse_transform(rede.predict(sx.transform(X_te), verbose=0)).ravel()
        st.session_state["tf_res"] = {"hist": hist.history, "r2": r2_score(y_te_, pred), "mae": mean_absolute_error(y_te_, pred)}
    if "tf_res" in st.session_state:
        res_tf = st.session_state["tf_res"]
        ref = avaliar_modelos(0.25, 100)[0]
        m1, m2, m3 = st.columns(3)
        m1.metric("R² teste (rede)", f"{res_tf['r2']:.3f}")
        m2.metric("MAE (kg/h)", f"{res_tf['mae']:.1f}")
        m3.metric("R² teste (Random Forest)", f"{ref.loc['Random Forest', 'R² teste']:.3f}")
        dfh = pd.DataFrame({"treino": res_tf["hist"]["loss"], "validação": res_tf["hist"]["val_loss"]})
        fig = px.line(dfh, labels={"index": "Época", "value": "Loss (MSE padronizado)", "variable": ""}, title="Curva de aprendizado")
        st.plotly_chart(fig)
        st.caption("Se a validação sobe enquanto o treino desce: *overfitting*. Ajuste épocas e neurônios.")

    st.code(
        '''
model = tf.keras.Sequential([
    tf.keras.Input(shape=(3,)),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse")
model.fit(X_tr_padronizado, y_tr_padronizado, validation_split=0.2, epochs=60)
''',
        language="python",
    )
    st.markdown("**Casos aeroespaciais típicos:**")
    st.markdown(
        "- **CNN**: detecção de trincas em imagens de inspeção ou de anomalias em imagens de satélite\n"
        "- **LSTM/GRU**: estimativa de vida útil restante (RUL) com séries de sensores — dataset NASA C-MAPSS\n"
        "- **Autoencoders**: detecção de anomalias em telemetria sem exemplos de falha"
    )
    st.subheader("🎯 Atividade")
    st.markdown(
        "1. Treine com 5 e com 200 épocas. Em qual a validação piora?\n"
        "2. A rede supera o Random Forest aqui? Por que isso é esperado com 400 linhas?\n"
        "3. Que tipo de rede você usaria para prever a RUL a partir de 21 sensores ao longo de 200 ciclos?"
    )

# =====================================================================
# ABA 6 – NLTK
# =====================================================================
with abas[6]:
    st.header("NLTK: texto como dado (diários de manutenção)")
    _, fonte_stop = carregar_stopwords()
    st.caption(f"Stopwords: {fonte_stop}. Para baixar antes da aula: `python -m nltk.downloader stopwords`")
    todos = [t for lst in LOGS.values() for t in lst]
    st.markdown("**Exemplos de registros (fictícios):**")
    st.dataframe(pd.DataFrame({"Registro": todos, "Sistema": [k for k, lst in LOGS.items() for _ in lst]}).sample(6, random_state=1), hide_index=True)

    filtro = st.multiselect("Sistemas a analisar", list(LOGS), default=list(LOGS))
    corpus = [t for k in filtro for t in LOGS[k]]
    tokens = [tk for texto in corpus for tk in preprocessar(texto)]
    if tokens:
        freq = FreqDist(tokens)
        top = pd.DataFrame(freq.most_common(12), columns=["palavra", "frequência"])
        st.plotly_chart(px.bar(top, x="frequência", y="palavra", orientation="h", title="Palavras mais frequentes").update_yaxes(autorange="reversed"))
    else:
        st.info("Selecione ao menos um sistema.")
    st.code(
        '''
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk import FreqDist

tok = RegexpTokenizer(r"[a-zA-ZÀ-ÿ]+")
stops = set(stopwords.words("portuguese"))
tokens = [t for t in tok.tokenize(texto.lower()) if t not in stops and len(t) > 2]
FreqDist(tokens).most_common(10)
''',
        language="python",
    )

    st.subheader("Classificador de registros: TF-IDF + Regressão Logística")
    pipe, acc = treinar_classificador()
    st.caption(f"Acurácia em validação cruzada (4 partes): {acc:.0%} — dataset minúsculo e ilustrativo!")
    texto = st.text_input("Digite um novo registro", "Óleo vazando próximo ao atuador do trem de pouso")
    if texto.strip():
        prob = pd.Series(pipe.predict_proba([texto])[0], index=pipe.classes_)
        st.success(f"Sistema previsto: **{prob.idxmax()}**")
        st.bar_chart(prob)

    st.subheader("🎯 Atividade")
    st.markdown(
        "1. Escreva um registro ambíguo (ex.: “vazamento de óleo no motor”). Que sistema o modelo escolhe? Com que confiança?\n"
        "2. Por que remover *stopwords* ajuda? Quando poderia atrapalhar? (pense em “não”)\n"
        "3. **Extensão:** adicione 5 registros novos por categoria e veja a acurácia mudar."
    )

# =====================================================================
# ABA 7 – PROJETO INTEGRADOR
# =====================================================================
with abas[7]:
    st.header("🏁 Projeto integrador: o ciclo completo")
    st.markdown("**Tema:** estimar a vida útil restante (RUL) de turbofans com o dataset **NASA C-MAPSS (FD001)**.")
    st.dataframe(
        pd.DataFrame(
            [
                ("1. Coleta", "Baixar e carregar o FD001 (txt separado por espaços)", "pandas"),
                ("2. Tratamento", "Criar a coluna RUL; remover sensores constantes; normalizar", "pandas, scikit-learn"),
                ("3. Modelagem", "Baseline (Regressão Linear/Random Forest) e comparar com uma rede (opcional)", "scikit-learn, TensorFlow"),
                ("4. Visualização", "RUL real vs. previsto; sensores ao longo da vida do motor", "Plotly / Matplotlib"),
            ],
            columns=["Etapa", "Tarefa", "Ferramentas"],
        ),
        hide_index=True,
    )
    st.code(
        '''
colunas = ["unidade", "ciclo", "ajuste_1", "ajuste_2", "ajuste_3"] + [f"sensor_{i}" for i in range(1, 22)]
df = pd.read_csv("train_FD001.txt", sep=r"\\s+", header=None, names=colunas)
df["RUL"] = df.groupby("unidade")["ciclo"].transform("max") - df["ciclo"]
''',
        language="python",
    )
    st.markdown(
        "**Entregáveis:** notebook/script reprodutível, 2 figuras interativas, e um parágrafo respondendo: "
        "*o modelo é confiável o bastante para apoiar uma decisão de manutenção? Por quê?*"
    )
    st.markdown("**Critérios:** reprodutibilidade · separação treino/teste correta · métricas adequadas · clareza das figuras.")