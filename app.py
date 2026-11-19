"""
Plataforma inteligente para la planeación de la producción y la distribución.
Analítica predictiva (ETL + pronósticos) y prescriptiva (optimización + escenarios).

Ejecutar:  streamlit run app.py
"""

import hashlib
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import datos
import escenarios as esc
import etl
import graficos as gr
import optimizacion as opt
import pronosticos as pr
from etl import mes_texto
from graficos import miles, pesos

st.set_page_config(
    page_title="Planeación de Producción y Distribución",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

RUTA = os.path.dirname(os.path.abspath(__file__))
RUTA_EJEMPLO = os.path.join(RUTA, "insumos", "Datos.xlsx")
CONFIG_GRAFICO = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}

# ---------------------------------------------------------------------------
# Estilo
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stDataFrame, button, input, textarea, select {
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
}
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"],
.stDeployButton, [data-testid="stAppDeployButton"], [data-testid="stStatusWidget"] {
    display: none !important; visibility: hidden !important;
}
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1380px; }
[data-testid="stSidebar"] { background: #0d2a4f; }
[data-testid="stSidebar"] * { color: #e9eef6; }
[data-testid="stSidebar"] .stRadio label p { font-size: 0.95rem; }
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] { background: #143a68; border-color: #3d6aa3; }
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] *:not(button):not(button *) { color: #e9eef6 !important; }
[data-testid="stSidebar"] [data-testid^="stBaseButton-secondary"] {
    background: #ffffff !important; border: 1px solid #ffffff !important; color: #0d2a4f !important; }
[data-testid="stSidebar"] [data-testid^="stBaseButton-secondary"] * { color: #0d2a4f !important; }
[data-testid="stSidebar"] [data-testid^="stBaseButton-secondary"]:hover { background: #dbe8f8 !important; }
[data-testid="stSidebar"] [data-testid^="stNumberInputStep"], [data-testid="stSidebar"] [data-testid^="stNumberInputStep"] * {
    color: #0d2a4f !important; }
[data-testid="stSidebar"] [data-testid="stBaseButton-headerNoPadding"] * { color: #e9eef6 !important; }
[data-testid="stSidebar"] [data-testid="stExpander"] { border-color: #2c5383; }
[data-testid="stSidebar"] input { color: #0b0b0b !important; }
[data-testid="stSidebar"] hr { border-color: #2c5383; }
.marca { font-size: 1.15rem; font-weight: 700; letter-spacing: .2px; margin: .2rem 0 0 0; }
.marca-sub { font-size: .78rem; opacity: .75; margin-bottom: .6rem; }
.hero {
    background: linear-gradient(120deg, #0d366b 0%, #1c5cab 55%, #2a78d6 100%);
    border-radius: 18px; padding: 1.6rem 1.9rem; color: #fff; margin-bottom: 1.2rem;
    box-shadow: 0 8px 24px rgba(13, 54, 107, .18);
}
.hero h1 { color: #fff; font-size: 1.75rem; font-weight: 700; margin: 0 0 .35rem 0; line-height: 1.25; }
.hero p { color: #dbe8f8; margin: 0; font-size: .98rem; }
.hero .chips { margin-top: .9rem; display: flex; flex-wrap: wrap; gap: .45rem; }
.hero .chip { background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.25);
    border-radius: 999px; padding: .22rem .75rem; font-size: .8rem; color: #fff; }
.titulo { font-size: 1.45rem; font-weight: 700; color: #0b0b0b; margin: .2rem 0 .1rem 0; }
.subtitulo { color: #52514e; font-size: .95rem; margin-bottom: 1rem; }
.kpi { background: #fff; border: 1px solid #e6e4dd; border-radius: 14px; padding: .95rem 1.1rem;
    box-shadow: 0 1px 2px rgba(0,0,0,.04); height: 100%; }
.kpi .lbl { color: #6b6a65; font-size: .78rem; font-weight: 600; text-transform: uppercase; letter-spacing: .4px; }
.kpi .val { color: #0b0b0b; font-size: 1.55rem; font-weight: 700; margin-top: .25rem; line-height: 1.2; }
.kpi .det { color: #52514e; font-size: .82rem; margin-top: .2rem; }
.kpi.acento { border-left: 4px solid #2a78d6; }
.paso { background: #fff; border: 1px solid #e6e4dd; border-radius: 14px; padding: 1rem 1.1rem; height: 100%; }
.paso .num { display: inline-block; background: #e8f1fc; color: #1c5cab; font-weight: 700; border-radius: 8px;
    padding: .1rem .55rem; font-size: .8rem; margin-bottom: .4rem; }
.paso h4 { margin: .1rem 0 .35rem 0; font-size: 1.02rem; color: #0b0b0b; }
.paso p { margin: 0; color: #52514e; font-size: .88rem; }
.estado { border-radius: 12px; padding: .8rem 1rem; margin: .3rem 0 1rem 0; font-size: .93rem; }
.estado.ok { background: #e9f7ec; border: 1px solid #b7e2c0; color: #11521f; }
.estado.alerta { background: #fff6e0; border: 1px solid #f5d98f; color: #6b4a00; }
.estado.error { background: #fdeceb; border: 1px solid #f2b8b5; color: #7a1d1b; }
.tarjeta-esc { background: #fff; border: 1px solid #e6e4dd; border-radius: 14px; padding: .9rem 1.05rem; height: 100%; }
.tarjeta-esc .nom { font-weight: 700; font-size: 1.02rem; color: #0b0b0b; }
.tarjeta-esc .desc { color: #6b6a65; font-size: .8rem; margin-bottom: .45rem; }
.tarjeta-esc .cifra { font-size: 1.3rem; font-weight: 700; color: #0b0b0b; }
.tarjeta-esc .linea { font-size: .84rem; color: #52514e; margin-top: .15rem; }
.etq { display: inline-block; border-radius: 999px; padding: .1rem .6rem; font-size: .74rem; font-weight: 600; }
.etq.ok { background: #e3f4e6; color: #11521f; }
.etq.alerta { background: #fff1cc; color: #6b4a00; }
.etq.error { background: #fbe0de; color: #7a1d1b; }
.nota { color: #6b6a65; font-size: .84rem; }
.respuesta { background: #fff; border: 1px solid #e6e4dd; border-radius: 12px; padding: .7rem .95rem; margin-bottom: .55rem; }
.respuesta .q { font-weight: 600; color: #1c5cab; font-size: .86rem; }
.respuesta .a { color: #0b0b0b; font-size: .92rem; margin-top: .15rem; }
div[data-testid="stTabs"] button p { font-size: .93rem; font-weight: 600; }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Utilidades de presentación
# ---------------------------------------------------------------------------
def encabezado(titulo, subtitulo=""):
    st.markdown(f'<div class="titulo">{titulo}</div><div class="subtitulo">{subtitulo}</div>',
                unsafe_allow_html=True)


def kpi(col, etiqueta, valor, detalle="", acento=False):
    largo = ' style="font-size:1.08rem"' if len(str(valor)) > 16 else ""
    col.markdown(
        f'<div class="kpi{" acento" if acento else ""}"><div class="lbl">{etiqueta}</div>'
        f'<div class="val"{largo}>{valor}</div><div class="det">{detalle}</div></div>',
        unsafe_allow_html=True,
    )


def estado(texto, tipo="ok"):
    st.markdown(f'<div class="estado {tipo}">{texto}</div>', unsafe_allow_html=True)


def grafico(fig, key=None):
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    fig.update_layout(template="plataforma")
    barras = [t for t in fig.data if t.type == "bar"]
    for t in barras:
        t.cliponaxis = False
    if any(t.orientation == "h" for t in barras):
        fig.update_layout(margin=dict(r=max(fig.layout.margin.r or 0, 80)))
    st.plotly_chart(fig, theme=None, config=CONFIG_GRAFICO, key=key)


def mostrar_tabla(df, dec=None, alto=None, estilo=None):
    dec = dec or {}
    fmt = {}
    for c in df.columns:
        serie = df[c]
        if pd.api.types.is_bool_dtype(serie):
            fmt[c] = lambda v: "Sí" if v else "No"
        elif pd.api.types.is_numeric_dtype(serie):
            if c in dec:
                d = dec[c]
            else:
                vals = serie.dropna()
                d = 0 if (len(vals) == 0 or np.allclose(vals % 1, 0)) else 2
            fmt[c] = (lambda d_: (lambda v: miles(v, d_)))(d)
        elif pd.api.types.is_datetime64_any_dtype(serie):
            fmt[c] = lambda v: v.strftime("%Y-%m-%d") if pd.notna(v) else "–"
    sty = df.style.format(fmt, na_rep="–")
    if estilo is not None:
        sty = estilo(sty)
    st.dataframe(sty, hide_index=True, height=alto if alto else "auto")


def descargar(df, nombre, etiqueta):
    st.download_button(etiqueta, df.to_csv(index=False).encode("utf-8-sig"), file_name=nombre,
                       mime="text/csv")


# ---------------------------------------------------------------------------
# Carga y cálculos (con caché)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def cargar_parametros(contenido):
    return datos.leer_parametros(contenido)


@st.cache_data(show_spinner=False)
def cargar_demanda(contenido_param, contenido_dem, nombre_dem):
    if contenido_dem is not None:
        return datos.leer_demanda(contenido_dem, nombre_dem), f"archivo cargado «{nombre_dem}»"
    df, hoja = datos.demanda_desde_libro(contenido_param)
    if df is not None:
        return df, f"hoja «{hoja}» de Datos.xlsx"
    return datos.generar_base_historica(), "base histórica de la Fase 1 (36 meses, semilla 42)"


@st.cache_data(show_spinner=False)
def procesar(plana):
    base, resumen, validacion, faltantes = etl.ejecutar_etl(plana)
    consolidada = etl.consolidar(base)
    return base, resumen, validacion, faltantes, consolidada, etl.base_modelos(consolidada)


@st.cache_data(show_spinner=False, persist="disk")
def pronosticar(consolidada):
    return pr.evaluar_todo(consolidada)


@st.cache_data(show_spinner=False)
def correr_escenario(nombre, demanda_base, clave, _parametros, ajuste, permitir, penal, cp=1.0, cn=1.0, ct=1.0):
    return esc.ejecutar_escenario(nombre, demanda_base, _parametros, ajuste, permitir_faltantes=permitir,
                                  factor_penalizacion=penal, factor_cap_plantas=cp, factor_cap_nodos=cn,
                                  factor_cap_transporte=ct)


@st.cache_data(show_spinner=False)
def descomponer(serie_valores, fechas):
    from statsmodels.tsa.seasonal import seasonal_decompose

    s = pd.Series(serie_valores, index=pd.DatetimeIndex(fechas, freq="MS"))
    return seasonal_decompose(s, model="multiplicative", period=12)


# ---------------------------------------------------------------------------
# Barra lateral
# ---------------------------------------------------------------------------
SECCIONES = [
    "Resumen general",
    "Datos y ETL",
    "Análisis descriptivo",
    "Pronósticos",
    "Optimización",
    "Escenarios",
    "Parámetros de la red",
]

with st.sidebar:
    st.markdown('<div class="marca">Planeación Inteligente</div>'
                '<div class="marca-sub">Producción · Trasbordo · Distribución</div>', unsafe_allow_html=True)
    archivo = st.file_uploader("Datos de la red (Datos.xlsx)", type=["xlsx", "xls"])
    with st.expander("Demanda histórica (opcional)"):
        archivo_dem = st.file_uploader(
            "CSV o Excel con Fecha, Producto, Cliente y Demanda", type=["csv", "xlsx", "xls"], key="dem")
        st.caption("Si no se carga, se usa la hoja de demanda del libro o, en su defecto, "
                   "la base histórica construida en la Fase 1.")
    if archivo is None and os.path.exists(RUTA_EJEMPLO):
        if st.session_state.get("usar_ejemplo"):
            st.caption("Usando el Datos.xlsx incluido en la carpeta insumos.")
        elif st.button("Usar el Datos.xlsx incluido", use_container_width=True):
            st.session_state["usar_ejemplo"] = True
            st.rerun()
    st.divider()
    seccion = st.radio("Secciones", SECCIONES, label_visibility="collapsed")
    st.divider()
    with st.expander("Opciones del modelo de optimización"):
        permitir_faltantes = st.toggle(
            "Permitir faltantes con penalización", value=False,
            help="Desactivado: toda la demanda debe atenderse y, si no hay capacidad, el modelo se reporta "
                 "como infactible. Activado: se admite demanda no atendida con un costo de penalización.")
        factor_penal = st.number_input(
            "Penalización por unidad no atendida (veces el precio base)", min_value=1.0, max_value=10.0,
            value=2.0, step=0.5, disabled=not permitir_faltantes)
    st.markdown('<div class="marca-sub" style="margin-top:1.2rem">Universidad Industrial de Santander<br>'
                'Analítica predictiva y prescriptiva · 2026</div>', unsafe_allow_html=True)


def hero(titulo, texto, chips=()):
    chips_html = "".join(f'<span class="chip">{c}</span>' for c in chips)
    st.markdown(f'<div class="hero"><h1>{titulo}</h1><p>{texto}</p><div class="chips">{chips_html}</div></div>',
                unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sin datos: pantalla de bienvenida
# ---------------------------------------------------------------------------
contenido = None
if archivo is not None:
    contenido = archivo.getvalue()
elif st.session_state.get("usar_ejemplo") and os.path.exists(RUTA_EJEMPLO):
    with open(RUTA_EJEMPLO, "rb") as fh:
        contenido = fh.read()

if contenido is None:
    hero("Plataforma inteligente para la planeación de la producción y la distribución",
         "Pronóstico de demanda por producto y cliente, y plan de producción, trasbordo y distribución "
         "de mínimo costo.",
         ["10 productos", "5 clientes", "4 plantas", "2 nodos de trasbordo", "4 métodos de pronóstico"])
    c1, c2, c3 = st.columns(3)
    pasos = [
        ("1", "Cargue Datos.xlsx", "Use el panel izquierdo para subir el libro con productos, clientes, plantas, "
                                   "nodos, capacidades y costos."),
        ("2", "Revise la analítica predictiva", "ETL, análisis descriptivo, cuatro modelos de pronóstico y "
                                                 "selección automática del mejor por serie."),
        ("3", "Tome decisiones", "Plan óptimo de producción y distribución, y comparación de los escenarios "
                                 "base, optimista y pesimista."),
    ]
    for col, (n, t, p) in zip((c1, c2, c3), pasos):
        col.markdown(f'<div class="paso"><span class="num">Paso {n}</span><h4>{t}</h4><p>{p}</p></div>',
                     unsafe_allow_html=True)
    st.write("")
    st.info("Hojas que debe contener el archivo: " + ", ".join(datos.HOJAS_REQUERIDAS) + ".")
    st.stop()

# ---------------------------------------------------------------------------
# Carga de datos y cálculos principales
# ---------------------------------------------------------------------------
try:
    parametros = cargar_parametros(contenido)
except Exception as e:
    st.error(f"No fue posible leer el archivo de parámetros: {e}")
    st.stop()
clave_param = hashlib.md5(contenido).hexdigest()

try:
    plana, fuente_demanda = cargar_demanda(
        contenido, archivo_dem.getvalue() if archivo_dem else None, archivo_dem.name if archivo_dem else "")
except Exception as e:
    st.error(f"No fue posible leer la demanda histórica: {e}")
    st.stop()

base, resumen_etl, validacion_etl, faltantes_col, consolidada, base_mod = procesar(plana)

with st.spinner("Entrenando y evaluando los cuatro modelos en las 50 series producto-cliente. "
                "La primera vez toma cerca de medio minuto…"):
    pruebas, metricas, pron_modelos, pron_final = pronosticar(consolidada)

fecha_pron = pd.Timestamp(pron_final["Fecha"].iloc[0])
mes_pron = mes_texto(fecha_pron, largo=True)

try:
    demanda_base = opt.preparar_demanda(pron_final, parametros)
except ValueError as e:
    st.error(str(e))
    st.stop()

resultados = {
    nombre: correr_escenario(nombre, demanda_base, clave_param, parametros, cfg["ajuste"],
                             permitir_faltantes, factor_penal)
    for nombre, cfg in esc.ESCENARIOS.items()
}
res_base = resultados["Base"]
resumen_modelos = pr.resumen_global(metricas)
mejor_global = resumen_modelos.iloc[0]


# ===========================================================================
# 1. RESUMEN GENERAL
# ===========================================================================
def pagina_resumen():
    hero("Plataforma inteligente para la planeación de la producción y la distribución",
         f"Analítica predictiva y prescriptiva para el horizonte de {mes_pron}: cuánto demandará cada cliente "
         "y cómo producir y distribuir al menor costo.",
         [f"{base['Producto'].nunique()} productos", f"{base['Cliente'].nunique()} clientes",
          f"{len(parametros['plantas'])} plantas", f"{len(parametros['nodos'])} nodos de trasbordo",
          f"{base['Fecha'].nunique()} meses de historia"])

    c = st.columns(5)
    kpi(c[0], "Demanda histórica", miles(base["Demanda"].sum()),
        f"unidades · {mes_texto(base['Fecha'].min())} a {mes_texto(base['Fecha'].max())}")
    kpi(c[1], f"Pronóstico {mes_pron}", miles(pron_final["Pronóstico"].sum()),
        f"unidades · {len(pron_final)} series producto-cliente", acento=True)
    kpi(c[2], "Mejor modelo global", mejor_global["Modelo"], f"MAPE {miles(mejor_global['MAPE'], 2)} %")
    if res_base["factible"]:
        kpi(c[3], "Costo mínimo (base)", pesos(res_base["costo_total"], millones=True),
            f"{pesos(res_base['costo_total'] / res_base['demanda_total'])} por unidad", acento=True)
        servicio = (1 - res_base["faltante_total"] / res_base["demanda_total"]) * 100
        kpi(c[4], "Nivel de servicio", f"{miles(servicio, 1)} %", "demanda atendida en el escenario base")
    else:
        kpi(c[3], "Costo mínimo (base)", "Infactible", "la capacidad no alcanza")
        kpi(c[4], "Nivel de servicio", "–", "revise la sección Escenarios")

    st.write("")
    st.markdown("##### Flujo de integración entre las dos analíticas")
    p = st.columns(3)
    pasos = [
        ("Analítica predictiva", "Demanda histórica → ETL → pronósticos (promedio móvil, Holt, Holt-Winters "
                                 "y MLP) → selección del mejor modelo por serie."),
        ("Analítica prescriptiva", "Demanda pronosticada + costos + capacidades → modelo de programación "
                                   "lineal de costo mínimo (PuLP)."),
        ("Decisiones óptimas", "Producción por planta, trasbordo por nodo, distribución a cada cliente "
                               "y costo total mínimo, evaluados en tres escenarios."),
    ]
    for i, (col, (t, d)) in enumerate(zip(p, pasos), start=1):
        col.markdown(f'<div class="paso"><span class="num">{i}</span><h4>{t}</h4><p>{d}</p></div>',
                     unsafe_allow_html=True)

    st.write("")
    izq, der = st.columns([1.55, 1])
    with izq:
        st.markdown("##### Red de suministro")
        st.caption("Pase el cursor sobre las rutas para ver distancia y capacidad. La ruta Cali → Nodo Norte "
                   "no está habilitada.")
        grafico(gr.red_logistica(parametros, alto=470))
    with der:
        st.markdown(f"##### Respuestas del plan óptimo · {mes_pron}")
        if not res_base["factible"]:
            estado("El escenario base no es factible con las capacidades actuales. "
                   "Revise la sección Escenarios.", "error")
        else:
            prod = res_base["produccion"]
            por_prod = prod.groupby("Producto")["Cantidad"].sum().sort_values(ascending=False)
            por_planta = res_base["plantas"].sort_values("Cantidad", ascending=False)
            nodos = res_base["nodos"]
            clientes = res_base["clientes"].sort_values("Entregado", ascending=False)
            respuestas = [
                ("¿Qué producir?", f"Los {(por_prod > 0.5).sum()} productos; el de mayor volumen es "
                                  f"{por_prod.index[0]} ({miles(por_prod.iloc[0])} u)."),
                ("¿Cuánto producir?", f"{miles(prod['Cantidad'].sum())} unidades en total."),
                ("¿En cuál planta?", " · ".join(f"{r.Planta.replace('Planta ', '')}: {miles(r.Cantidad)} u"
                                                for r in por_planta.itertuples())),
                ("¿Cuánto por cada nodo?", " · ".join(f"{r['Nodo']}: {miles(r['Cantidad'])} u "
                                                      f"({miles(r['Utilización (%)'], 1)} % de su capacidad)"
                                                      for _, r in nodos.iterrows())),
                ("¿Cuánto a cada cliente?", " · ".join(f"{r.Cliente}: {miles(r.Entregado)} u"
                                                       for r in clientes.itertuples())),
                ("¿Costo total mínimo?", f"{pesos(res_base['costo_total'])} COP"),
            ]
            for q, a in respuestas:
                st.markdown(f'<div class="respuesta"><div class="q">{q}</div><div class="a">{a}</div></div>',
                            unsafe_allow_html=True)


# ===========================================================================
# 2. DATOS Y ETL
# ===========================================================================
def pagina_etl():
    encabezado("Datos y proceso ETL",
               "Extracción, limpieza, verificación y consolidación de la demanda por producto, cliente y período.")
    estado(f"Parámetros de la red leídos de <b>Datos.xlsx</b> ({len(parametros['libro'])} hojas). "
           f"Demanda histórica tomada de la <b>{fuente_demanda}</b>.", "ok")

    r = dict(zip(resumen_etl["Indicador"], resumen_etl["Resultado"]))
    c = st.columns(5)
    kpi(c[0], "Registros originales", miles(r["Registros originales"]))
    kpi(c[1], "Duplicados eliminados", miles(r["Registros duplicados eliminados"]))
    kpi(c[2], "Datos faltantes", miles(r["Datos faltantes iniciales"]), "imputados con la mediana del producto")
    kpi(c[3], "Atípicos en demanda", miles(r["Atípicos en demanda (IQR, conservados)"]),
        "marcados por IQR y conservados")
    kpi(c[4], "Registros limpios", miles(r["Registros finales limpios"]), "listos para los modelos", acento=True)
    st.write("")

    t1, t2, t3 = st.tabs(["Calidad de los datos", "Base limpia", "Bases para los modelos"])
    with t1:
        a, b = st.columns(2)
        with a:
            st.markdown("##### Resumen del ETL")
            mostrar_tabla(resumen_etl)
            st.markdown("##### Faltantes por variable (antes de imputar)")
            f = faltantes_col.reset_index()
            f.columns = ["Variable", "Faltantes"]
            mostrar_tabla(f)
        with b:
            st.markdown("##### Consistencia de fechas y cantidades")
            mostrar_tabla(validacion_etl)
            st.markdown("##### Atípicos de demanda por producto")
            at = base.groupby("Producto")["Atipico_Demanda"].sum().astype(int).reset_index()
            at.columns = ["Producto", "Registros atípicos"]
            mostrar_tabla(at.sort_values("Registros atípicos", ascending=False))
        fig = px.box(base, x="Producto", y="Demanda", points="outliers",
                     color_discrete_sequence=[gr.SERIE[0]])
        fig.update_traces(marker=dict(color=gr.SERIE[1], size=6), line=dict(width=1.5, color=gr.SERIE[0]),
                          fillcolor="rgba(42,120,214,0.18)")
        fig.update_xaxes(title=None, tickangle=-25)
        fig.update_yaxes(title="Demanda mensual por cliente (u)")
        grafico(gr.layout(fig, "Distribución de la demanda por producto y valores atípicos (IQR)", 430))
    with t2:
        f1, f2 = st.columns(2)
        sel_p = f1.multiselect("Producto", sorted(base["Producto"].unique()))
        sel_c = f2.multiselect("Cliente", sorted(base["Cliente"].unique()))
        vista = base.copy()
        if sel_p:
            vista = vista[vista["Producto"].isin(sel_p)]
        if sel_c:
            vista = vista[vista["Cliente"].isin(sel_c)]
        st.caption(f"{miles(len(vista))} registros")
        mostrar_tabla(vista, alto=420)
        descargar(base, "base_limpia.csv", "Descargar base limpia (CSV)")
    with t3:
        st.markdown("##### Demanda consolidada por producto, cliente y período")
        mostrar_tabla(consolidada.head(300), alto=300)
        descargar(consolidada, "demanda_producto_cliente_periodo.csv", "Descargar consolidado (CSV)")
        st.markdown("##### Base para los modelos predictivos (variables temporales, cíclicas y rezagos)")
        mostrar_tabla(base_mod.head(300), dec={"Mes_sin": 4, "Mes_cos": 4}, alto=300)
        descargar(base_mod, "base_modelos.csv", "Descargar base para modelos (CSV)")


# ===========================================================================
# 3. ANÁLISIS DESCRIPTIVO
# ===========================================================================
def pagina_descriptiva():
    encabezado("Análisis descriptivo de la demanda",
               "Comportamiento histórico, participación, tendencia, estacionalidad y variabilidad.")
    mensual = base.groupby("Fecha", as_index=False)["Demanda"].sum()
    por_prod = base.groupby("Producto", as_index=False)["Demanda"].sum().sort_values("Demanda", ascending=False)
    por_cli = base.groupby("Cliente", as_index=False)["Demanda"].sum().sort_values("Demanda", ascending=False)
    por_cli["Participación (%)"] = por_cli["Demanda"] / por_cli["Demanda"].sum() * 100

    c = st.columns(4)
    kpi(c[0], "Demanda mensual promedio", miles(mensual["Demanda"].mean()), "unidades por mes")
    kpi(c[1], "Producto con mayor demanda", por_prod.iloc[0]["Producto"], f"{miles(por_prod.iloc[0]['Demanda'])} u")
    kpi(c[2], "Producto con menor demanda", por_prod.iloc[-1]["Producto"], f"{miles(por_prod.iloc[-1]['Demanda'])} u")
    kpi(c[3], "Cliente con mayor participación", por_cli.iloc[0]["Cliente"],
        f"{miles(por_cli.iloc[0]['Participación (%)'], 1)} % de la demanda")
    st.write("")

    t1, t2, t3, t4 = st.tabs(["Comportamiento histórico", "Productos y clientes",
                              "Tendencia y estacionalidad", "Variabilidad"])
    with t1:
        mensual["Media móvil 3 meses"] = mensual["Demanda"].rolling(3).mean()
        fig = go.Figure()
        fig.add_scatter(x=mensual["Fecha"], y=mensual["Demanda"], name="Demanda total", mode="lines+markers",
                        line=dict(color=gr.SERIE[0], width=2), marker=dict(size=7))
        fig.add_scatter(x=mensual["Fecha"], y=mensual["Media móvil 3 meses"], name="Media móvil de 3 meses",
                        mode="lines", line=dict(color=gr.TENUE, width=2, dash="dash"))
        fig.update_yaxes(title="Unidades", rangemode="normal")
        fig.update_layout(hovermode="x unified")
        gr.eje_meses(fig, mensual["Fecha"], 3)
        grafico(gr.layout(fig, "Demanda total mensual", 380))

        prod_mes = base.groupby(["Producto", "Fecha"], as_index=False)["Demanda"].sum()
        fig = px.line(prod_mes, x="Fecha", y="Demanda", facet_col="Producto", facet_col_wrap=5,
                      facet_row_spacing=0.14, facet_col_spacing=0.04, color_discrete_sequence=[gr.SERIE[0]])
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1], font=dict(size=11, color=gr.TINTA)))
        fig.update_yaxes(matches=None, showticklabels=True, title=None, rangemode="normal")
        fig.update_xaxes(title=None)
        gr.eje_meses(fig, prod_mes["Fecha"], 12)
        fig.update_traces(line=dict(width=2))
        grafico(gr.layout(fig, "Evolución mensual por producto (escala propia en cada panel)", 520))

        acum = por_prod.copy()
        acum["Participación (%)"] = acum["Demanda"] / acum["Demanda"].sum() * 100
        acum["Acumulado (%)"] = acum["Participación (%)"].cumsum()
        acum.columns = ["Producto", "Demanda acumulada (u)", "Participación (%)", "Participación acumulada (%)"]
        st.markdown("##### Demanda acumulada por producto en el horizonte histórico")
        mostrar_tabla(acum, dec={"Participación (%)": 1, "Participación acumulada (%)": 1})

    with t2:
        a, b = st.columns(2)
        with a:
            orden = por_prod.sort_values("Demanda")
            colores = [gr.SERIE[0] if i >= len(orden) - 3 else (gr.SERIE[1] if i < 3 else "#c9c7bf")
                       for i in range(len(orden))]
            fig = go.Figure(go.Bar(x=orden["Demanda"], y=orden["Producto"], orientation="h", marker_color=colores,
                                   text=[miles(v) for v in orden["Demanda"]], textposition="outside",
                                   hovertemplate="%{y}: %{x:,.0f} u<extra></extra>"))
            fig.update_xaxes(title="Unidades (36 meses)", showgrid=True, gridcolor=gr.REJILLA,
                             range=[0, orden["Demanda"].max() * 1.18])
            gr.barras_redondeadas(fig)
            grafico(gr.layout(fig, "Demanda total por producto", 430))
            st.caption("Azul: los tres productos de mayor demanda. Naranja: los tres de menor demanda.")
        with b:
            orden = por_cli.sort_values("Demanda")
            fig = go.Figure(go.Bar(x=orden["Participación (%)"], y=orden["Cliente"], orientation="h",
                                   marker_color=gr.SERIE[2],
                                   text=[f"{miles(p, 1)} % · {miles(v)} u" for p, v in
                                         zip(orden["Participación (%)"], orden["Demanda"])],
                                   textposition="outside", hovertemplate="%{y}: %{x:.1f} %<extra></extra>"))
            fig.update_xaxes(title="Participación en la demanda total (%)", showgrid=True, gridcolor=gr.REJILLA,
                             range=[0, orden["Participación (%)"].max() * 1.45])
            gr.barras_redondeadas(fig)
            grafico(gr.layout(fig, "Participación de los clientes", 430))
        piv = base.pivot_table(index="Producto", columns="Cliente", values="Demanda", aggfunc="sum")
        piv.columns = [gr.partir(c) for c in piv.columns]
        fig = px.imshow(piv, text_auto=",.0f", aspect="auto", color_continuous_scale=gr.AZUL_SEC)
        fig.update_layout(coloraxis_colorbar=dict(title="u"))
        fig.update_xaxes(title=None, side="top")
        fig.update_yaxes(title=None)
        grafico(gr.layout(fig, "Demanda histórica por producto y cliente (unidades)", 460, margin=dict(t=90)))

    with t3:
        prod_mes = base.groupby(["Producto", "Fecha"], as_index=False)["Demanda"].sum()
        prod_mes["Mes"] = prod_mes["Fecha"].dt.month
        indice = prod_mes.groupby(["Producto", "Mes"])["Demanda"].mean().unstack()
        indice = indice.div(indice.mean(axis=1), axis=0)
        indice.columns = [m.capitalize() for m in etl.MESES]
        fig = px.imshow(indice, text_auto=".2f", aspect="auto", zmin=0.3, zmax=1.7,
                        color_continuous_scale=[[0, "#e34948"], [0.5, "#f0efec"], [1, "#2a78d6"]])
        fig.update_layout(coloraxis_colorbar=dict(title="Índice"))
        fig.update_xaxes(title=None, side="top")
        fig.update_yaxes(title=None)
        grafico(gr.layout(fig, "Índice estacional por mes (1 = mes promedio; azul por encima, rojo por debajo)",
                          440, margin=dict(t=90)))

        # Tendencia lineal mensual
        filas = []
        for p_, g in prod_mes.groupby("Producto"):
            y = g.sort_values("Fecha")["Demanda"].to_numpy(dtype=float)
            pendiente = np.polyfit(np.arange(len(y)), y, 1)[0]
            cv_est = indice.loc[p_].std() * 100
            filas.append({"Producto": p_, "Tendencia (u/mes)": pendiente,
                          "Tendencia (% mensual)": pendiente / y.mean() * 100,
                          "Amplitud estacional (desv. índice, %)": cv_est})
        tend = pd.DataFrame(filas)
        tend["Patrón dominante"] = np.where(
            tend["Amplitud estacional (desv. índice, %)"] > 20, "Estacional",
            np.where(tend["Tendencia (% mensual)"] > 0.3, "Tendencia creciente",
                     np.where(tend["Tendencia (% mensual)"] < -0.3, "Tendencia decreciente", "Estable")))
        st.markdown("##### Tendencia y estacionalidad por producto")
        mostrar_tabla(tend.sort_values("Amplitud estacional (desv. índice, %)", ascending=False),
                      dec={"Tendencia (u/mes)": 1, "Tendencia (% mensual)": 2,
                           "Amplitud estacional (desv. índice, %)": 1})

        st.markdown("##### Descomposición multiplicativa (período de 12 meses)")
        sel = st.selectbox("Producto a descomponer", sorted(base["Producto"].unique()))
        g = prod_mes[prod_mes["Producto"] == sel].sort_values("Fecha")
        d = descomponer(g["Demanda"].to_numpy(dtype=float), list(g["Fecha"]))
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                            subplot_titles=("Observada", "Tendencia", "Estacionalidad", "Residuo"))
        comps = [(d.observed, gr.SERIE[0]), (d.trend, gr.SERIE[1]), (d.seasonal, gr.SERIE[2]), (d.resid, gr.TENUE)]
        for i, (s, col) in enumerate(comps, start=1):
            fig.add_scatter(x=s.index, y=s.values, mode="lines+markers" if i == 4 else "lines",
                            line=dict(color=col, width=2), marker=dict(size=4), showlegend=False, row=i, col=1)
            fig.update_yaxes(rangemode="normal", row=i, col=1)
        fig.for_each_annotation(lambda a: a.update(font=dict(size=12, color=gr.TINTA), x=0, xanchor="left"))
        gr.eje_meses(fig, g["Fecha"], 3)
        grafico(gr.layout(fig, None, 640, margin=dict(t=30)))

    with t4:
        vp = etl.variabilidad(base, "Producto")
        vc = etl.variabilidad(base, "Cliente")
        a, b = st.columns([1.1, 1])
        with a:
            orden = vp.sort_values("CV (%)")
            fig = go.Figure(go.Bar(x=orden["CV (%)"], y=orden["Producto"], orientation="h",
                                   marker_color=gr.SERIE[6], text=[f"{miles(v, 1)} %" for v in orden["CV (%)"]],
                                   textposition="outside", hovertemplate="%{y}: %{x:.1f} %<extra></extra>"))
            fig.update_xaxes(title="Coeficiente de variación (%)", showgrid=True, gridcolor=gr.REJILLA,
                             range=[0, orden["CV (%)"].max() * 1.2])
            gr.barras_redondeadas(fig)
            grafico(gr.layout(fig, "Variabilidad de la demanda mensual por producto", 420))
        with b:
            tot = mensual["Demanda"]
            st.markdown("##### Demanda total mensual")
            mostrar_tabla(pd.DataFrame({
                "Estadístico": ["Promedio", "Desviación estándar", "Mínimo", "Máximo", "Coef. de variación (%)"],
                "Valor": [tot.mean(), tot.std(), tot.min(), tot.max(), tot.std() / tot.mean() * 100]}),
                dec={"Valor": 1})
            st.caption("Un coeficiente de variación alto indica una demanda más difícil de anticipar; "
                       "en esta base lo explican sobre todo los productos estacionales.")
        st.markdown("##### Estadísticos por producto")
        mostrar_tabla(vp, dec={"Promedio": 1, "Desv. estándar": 1, "CV (%)": 1})
        st.markdown("##### Estadísticos por cliente")
        mostrar_tabla(vc, dec={"Promedio": 1, "Desv. estándar": 1, "CV (%)": 1})


# ===========================================================================
# 4. PRONÓSTICOS
# ===========================================================================
def pagina_pronosticos():
    encabezado("Modelos de pronóstico y selección automática",
               "Validación walk-forward en los últimos 6 meses: cada mes se pronostica solo con la información "
               "real disponible hasta el mes anterior. Se elige el modelo de menor MAPE para cada serie.")
    ini, fin = pruebas["Fecha"].min(), pruebas["Fecha"].max()
    conteo = pron_final["Modelo seleccionado"].value_counts()

    c = st.columns(4)
    for col, (_, r) in zip(c, resumen_modelos.iterrows()):
        gana = int(conteo.get(r["Modelo"], 0))
        kpi(col, r["Modelo"], f"MAPE {miles(r['MAPE'], 2)} %",
            f"MAE {miles(r['MAE'], 2)} · RMSE {miles(r['RMSE'], 2)}<br>Elegido en {gana} de {len(pron_final)} series",
            acento=r["Modelo"] == mejor_global["Modelo"])
    st.write("")

    t1, t2, t3 = st.tabs(["Comparación de modelos", "Explorar una serie", f"Pronóstico final · {mes_pron}"])
    with t1:
        st.caption(f"Período de validación: {mes_texto(ini)} a {mes_texto(fin)}. "
                   "Promedio móvil de 3 meses · Holt (nivel y tendencia) · Holt-Winters con tendencia aditiva y "
                   "estacionalidad multiplicativa (12 meses) · MLP con 6 rezagos, mes y tiempo, capas de 32 y 16 "
                   "neuronas ReLU, optimizador Adam y pérdida MSE.")
        a, b = st.columns(2)
        with a:
            rm = resumen_modelos.sort_values("MAPE")
            fig = go.Figure(go.Bar(x=rm["Modelo"], y=rm["MAPE"], marker_color=[gr.COLOR_MODELO[m] for m in rm["Modelo"]],
                                   text=[f"{miles(v, 2)} %" for v in rm["MAPE"]], textposition="outside",
                                   hovertemplate="%{x}: %{y:.2f} %<extra></extra>"))
            fig.update_yaxes(title="MAPE promedio (%)", range=[0, rm["MAPE"].max() * 1.2])
            gr.barras_redondeadas(fig)
            grafico(gr.layout(fig, "Error porcentual promedio de las 50 series", 380))
        with b:
            cnt = conteo.reindex(pr.MODELOS).fillna(0).sort_values(ascending=False)
            fig = go.Figure(go.Bar(x=cnt.index, y=cnt.values, marker_color=[gr.COLOR_MODELO[m] for m in cnt.index],
                                   text=[f"{int(v)} ({v / cnt.sum() * 100:.0f} %)" for v in cnt.values],
                                   textposition="outside", hovertemplate="%{x}: %{y} series<extra></extra>"))
            fig.update_yaxes(title="Series en que resulta el mejor", range=[0, cnt.max() * 1.2])
            gr.barras_redondeadas(fig)
            grafico(gr.layout(fig, "Modelo seleccionado por serie producto-cliente", 380))

        st.markdown("##### Tabla comparativa global")
        mostrar_tabla(resumen_modelos, dec={"MAE": 2, "RMSE": 2, "MAPE": 2})

        st.markdown("##### Error promedio por producto y modelo")
        metrica = st.segmented_control("Métrica", ["MAPE", "MAE", "RMSE"], default="MAPE",
                                       label_visibility="collapsed") or "MAPE"
        tp = pr.tabla_por_producto(metricas, metrica)
        mostrar_tabla(tp, dec={m: 2 for m in pr.MODELOS}, estilo=lambda s: s.highlight_min(
            subset=pr.MODELOS, axis=1, props="background-color:#dcebfb;color:#0d366b;font-weight:700;"))
        st.caption("La celda resaltada en cada fila es el menor error; la columna Ganador lo resume. "
                   "MAPE en %, MAE y RMSE en unidades.")

        fig = px.box(metricas, x="Modelo", y="MAPE", color="Modelo", color_discrete_map=gr.COLOR_MODELO,
                     category_orders={"Modelo": list(resumen_modelos["Modelo"])}, points="outliers",
                     hover_data=["Producto", "Cliente"])
        fig.update_yaxes(title="MAPE por serie (%)")
        fig.update_xaxes(title=None)
        grafico(gr.layout(fig, "Dispersión del MAPE en las 50 series", 400, showlegend=False))

    with t2:
        f1, f2 = st.columns(2)
        prods = sorted(consolidada["Producto"].unique())
        clis = sorted(consolidada["Cliente"].unique())
        p_sel = f1.selectbox("Producto", prods, index=prods.index("Cuadernos escolares") if "Cuadernos escolares" in prods else 0)
        c_sel = f2.selectbox("Cliente", clis, index=clis.index("Grupo Andino Retail") if "Grupo Andino Retail" in clis else 0)
        serie = consolidada[(consolidada["Producto"] == p_sel) & (consolidada["Cliente"] == c_sel)].sort_values("Fecha")
        pr_s = pruebas[(pruebas["Producto"] == p_sel) & (pruebas["Cliente"] == c_sel)]
        met_s = metricas[(metricas["Producto"] == p_sel) & (metricas["Cliente"] == c_sel)].sort_values("MAPE")
        fin_s = pron_modelos[(pron_modelos["Producto"] == p_sel) & (pron_modelos["Cliente"] == c_sel)]
        elegido = pron_final[(pron_final["Producto"] == p_sel) & (pron_final["Cliente"] == c_sel)].iloc[0]

        fig = go.Figure()
        fig.add_vrect(x0=ini - pd.Timedelta(days=15), x1=fecha_pron + pd.Timedelta(days=15),
                      fillcolor="#f2f1ec", opacity=1, layer="below", line_width=0)
        fig.add_annotation(x=ini, y=1.02, yref="paper", text="Validación y pronóstico", showarrow=False,
                           xanchor="left", font=dict(size=11, color=gr.TENUE))
        fig.add_scatter(x=serie["Fecha"], y=serie["Demanda"], name="Demanda real", mode="lines+markers",
                        line=dict(color=gr.TINTA_2, width=2), marker=dict(size=6))
        for m in pr.MODELOS:
            g = pr_s[pr_s["Modelo"] == m]
            principal = m == elegido["Modelo seleccionado"]
            fig.add_scatter(x=g["Fecha"], y=g["Pronóstico"], name=m + (" (seleccionado)" if principal else ""),
                            mode="lines+markers", legendgroup=m,
                            line=dict(color=gr.COLOR_MODELO[m], width=3 if principal else 1.6,
                                      dash="solid" if principal else "dot"),
                            marker=dict(size=8 if principal else 5))
            v = fin_s[fin_s["Modelo"] == m]["Pronóstico"]
            fig.add_scatter(x=[fecha_pron], y=v, mode="markers", legendgroup=m, showlegend=False,
                            name=f"{m} · {mes_pron}",
                            marker=dict(size=14 if principal else 9, color=gr.COLOR_MODELO[m],
                                        symbol="diamond" if principal else "circle",
                                        line=dict(color="white", width=2)))
        fig.update_yaxes(title="Unidades", rangemode="normal")
        fig.update_layout(hovermode="x unified")
        gr.eje_meses(fig, list(serie["Fecha"]) + [fecha_pron], 3)
        grafico(gr.layout(fig, f"{p_sel} · {c_sel}", 470))

        a, b = st.columns([1.3, 1])
        with a:
            st.markdown("##### Métricas de validación de la serie")
            t = met_s.drop(columns=["Producto", "Cliente"]).merge(
                fin_s[["Modelo", "Pronóstico"]], on="Modelo").rename(columns={"Pronóstico": f"Pronóstico {mes_pron}"})
            mostrar_tabla(t, dec={"MAE": 2, "RMSE": 2, "MAPE": 2, f"Pronóstico {mes_pron}": 1},
                          estilo=lambda s: s.apply(
                              lambda r: ["background-color:#dcebfb;font-weight:700" if r["Modelo"] ==
                                         elegido["Modelo seleccionado"] else "" for _ in r], axis=1))
        with b:
            kpi(st, "Modelo seleccionado", elegido["Modelo seleccionado"],
                f"MAPE {miles(elegido['MAPE'], 2)} % en validación", acento=True)
            st.write("")
            kpi(st, f"Pronóstico {mes_pron}", f"{miles(elegido['Pronóstico'], 1)} u",
                "insumo del modelo de optimización")

    with t3:
        tot = pron_final["Pronóstico"].sum()
        por_p = pron_final.groupby("Producto")["Pronóstico"].sum().sort_values()
        c = st.columns(3)
        kpi(c[0], f"Demanda pronosticada · {mes_pron}", miles(tot, 0), "unidades, suma de las 50 series", acento=True)
        kpi(c[1], "Producto con mayor pronóstico", por_p.index[-1], f"{miles(por_p.iloc[-1])} u")
        kpi(c[2], "MAPE promedio de los modelos elegidos", f"{miles(pron_final['MAPE'].mean(), 2)} %",
            f"{(pron_final['MAPE'] < 10).sum()} de {len(pron_final)} series por debajo del 10 %")
        st.write("")
        piv = pron_final.pivot_table(index="Producto", columns="Cliente", values="Pronóstico", aggfunc="sum")
        piv.columns = [gr.partir(c) for c in piv.columns]
        fig = px.imshow(piv, text_auto=",.0f", aspect="auto", color_continuous_scale=gr.AZUL_SEC)
        fig.update_layout(coloraxis_colorbar=dict(title="u"))
        fig.update_xaxes(title=None, side="top")
        fig.update_yaxes(title=None)
        grafico(gr.layout(fig, f"Demanda pronosticada por producto y cliente · {mes_pron} (unidades)", 470,
                          margin=dict(t=90)))
        fig = go.Figure(go.Bar(x=por_p.values, y=por_p.index, orientation="h", marker_color=gr.SERIE[0],
                               text=[miles(v) for v in por_p.values], textposition="outside",
                               hovertemplate="%{y}: %{x:,.0f} u<extra></extra>"))
        fig.update_xaxes(title="Unidades", showgrid=True, gridcolor=gr.REJILLA, range=[0, por_p.max() * 1.15])
        gr.barras_redondeadas(fig)
        grafico(gr.layout(fig, f"Pronóstico total por producto · {mes_pron}", 420))
        st.markdown("##### Detalle del pronóstico final")
        mostrar_tabla(pron_final, dec={"Pronóstico": 2, "MAE": 2, "RMSE": 2, "MAPE": 2}, alto=420)
        d1, d2, d3 = st.columns(3)
        with d1:
            descargar(pron_final, "pronostico_final.csv", "Descargar pronóstico final")
        with d2:
            descargar(metricas, "metricas_modelos.csv", "Descargar métricas por serie")
        with d3:
            descargar(pruebas, "resultados_validacion.csv", "Descargar pronósticos de validación")


# ===========================================================================
# 5. OPTIMIZACIÓN
# ===========================================================================
def mostrar_infactible(r):
    estado(f"<b>El modelo no es factible.</b> Con las capacidades actuales no es posible atender toda la "
           f"demanda de {miles(r['demanda_total'])} unidades. No se presenta un plan porque cualquier solución "
           "violaría alguna restricción. Active «Permitir faltantes con penalización» en el panel izquierdo "
           "para obtener el mejor plan posible con demanda no atendida.", "error")
    diag = r["diagnostico"]
    insuf = diag[diag["Estado"] == "Insuficiente"]
    if len(insuf):
        st.markdown("##### Recursos con capacidad insuficiente")
        mostrar_tabla(insuf, dec={"Uso requerido (%)": 1})
    aux = r.get("diagnostico_faltantes")
    if aux and aux.get("factible"):
        st.markdown(f"##### Demanda que no podría atenderse: {miles(aux['faltante_total'])} unidades")
        f = aux["atencion"]
        f = f[f["Faltante"] > 0.5][["Producto", "Cliente", "Demanda", "Entregado", "Faltante"]]
        mostrar_tabla(f.sort_values("Faltante", ascending=False))
        st.markdown("##### Restricciones que lo impiden (al 100 % de su capacidad)")
        mostrar_tabla(aux["activas"], dec={"Precio sombra (COP/u)": 0})


def pagina_optimizacion():
    encabezado(f"Plan óptimo de producción, trasbordo y distribución · {mes_pron}",
               "Programación lineal de costo mínimo con la demanda pronosticada (escenario base).")
    r = res_base
    if not r["factible"]:
        mostrar_infactible(r)
        return
    if r["faltante_total"] > 0.5:
        estado(f"Solución óptima con <b>{miles(r['faltante_total'])} unidades no atendidas</b> "
               f"(penalización de {miles(factor_penal, 1)} veces el precio base).", "alerta")
    else:
        estado(f"<b>Solución óptima encontrada.</b> Se atiende el 100 % de la demanda pronosticada "
               f"({miles(r['demanda_total'])} unidades) respetando todas las capacidades.", "ok")

    cst = r["costos"]
    c = st.columns(5)
    kpi(c[0], "Costo total mínimo", pesos(cst["Total operación"], millones=True),
        f"{pesos(cst['Total operación'])} COP", acento=True)
    kpi(c[1], "Producción", pesos(cst["Producción"], millones=True),
        f"{miles(cst['Producción'] / cst['Total operación'] * 100, 1)} % del total")
    kpi(c[2], "Transporte planta → nodo", pesos(cst["Transporte planta-nodo"], millones=True),
        f"{miles(cst['Transporte planta-nodo'] / cst['Total operación'] * 100, 1)} % del total")
    kpi(c[3], "Distribución nodo → cliente", pesos(cst["Distribución nodo-cliente"], millones=True),
        f"{miles(cst['Distribución nodo-cliente'] / cst['Total operación'] * 100, 1)} % del total")
    kpi(c[4], "Costo por unidad", pesos(cst["Total operación"] / max(r["demanda_total"] - r["faltante_total"], 1)),
        "COP por unidad entregada")
    st.write("")

    t1, t2, t3, t4, t5 = st.tabs(["Plan óptimo", "Plantas y nodos", "Distribución a clientes",
                                  "Restricciones activas", "Modelo matemático"])
    with t1:
        st.markdown("##### Flujo óptimo: plantas → nodos de trasbordo → clientes")
        grafico(gr.sankey(r["flujo_pn"], r["flujo_nc"], 480))
        prod = r["produccion"]
        piv = prod.pivot_table(index="Producto", columns="Planta", values="Cantidad", aggfunc="sum")
        orden_pl = list(r["plantas"]["Planta"])
        piv = piv.reindex(columns=[p for p in orden_pl if p in piv.columns])
        texto = piv.map(lambda v: "" if pd.isna(v) else miles(v))
        fig = go.Figure(go.Heatmap(z=piv.values, x=piv.columns, y=piv.index, colorscale=gr.AZUL_SEC,
                                   text=texto.values, texttemplate="%{text}", hoverongaps=False,
                                   hovertemplate="%{y} en %{x}: %{z:,.0f} u<extra></extra>",
                                   colorbar=dict(title="u")))
        fig.update_xaxes(side="top", showgrid=False)
        fig.update_yaxes(showgrid=False, autorange="reversed")
        grafico(gr.layout(fig, "¿Qué y cuánto producir en cada planta? (celdas vacías: la planta no fabrica ese producto)",
                          470, margin=dict(t=90)))
        st.markdown("##### Plan de producción")
        tabla = prod[prod["Cantidad"] > 0.5][["Planta", "Producto", "Cantidad", "Capacidad", "Utilización (%)"]]
        mostrar_tabla(tabla.sort_values(["Planta", "Cantidad"], ascending=[True, False]),
                      dec={"Utilización (%)": 1}, alto=380)
        descargar(prod, "plan_produccion.csv", "Descargar plan de producción")

    with t2:
        a, b = st.columns(2)
        with a:
            grafico(gr.barras_utilizacion(r["plantas"], "Planta", color_col="id_planta", colores=gr.COLOR_PLANTA,
                                          titulo="Utilización de la capacidad de las plantas", alto=330))
            mostrar_tabla(r["plantas"].drop(columns=["id_planta"]), dec={"Utilización (%)": 1})
        with b:
            grafico(gr.barras_utilizacion(r["nodos"], "Nodo", color_col="id_nodo", colores=gr.COLOR_NODO,
                                          titulo="Utilización de los nodos de trasbordo", alto=330))
            mostrar_tabla(r["nodos"].drop(columns=["id_nodo"]), dec={"Utilización (%)": 1})
        st.markdown("##### Uso de las rutas de transporte")
        rpn = r["flujo_pn"].groupby(["id_planta", "id_nodo", "Planta", "Nodo"], as_index=False)["Cantidad"].sum()
        rpn = rpn.merge(parametros["rutas_pn"], on=["id_planta", "id_nodo"])
        rpn["Utilización (%)"] = rpn["Cantidad"] / rpn["capacidad_transporte"] * 100
        rpn = rpn.rename(columns={"capacidad_transporte": "Capacidad", "distancia_km": "Distancia (km)"})
        rnc = r["flujo_nc"].groupby(["id_nodo", "id_cliente", "Nodo", "Cliente"], as_index=False)["Cantidad"].sum()
        rnc = rnc.merge(parametros["rutas_nc"], on=["id_nodo", "id_cliente"])
        rnc["Utilización (%)"] = rnc["Cantidad"] / rnc["capacidad_transporte"] * 100
        rnc = rnc.rename(columns={"capacidad_transporte": "Capacidad", "distancia_km": "Distancia (km)"})
        st.caption("Planta → nodo")
        mostrar_tabla(rpn[["Planta", "Nodo", "Distancia (km)", "Cantidad", "Capacidad", "Utilización (%)"]]
                      .sort_values("Utilización (%)", ascending=False), dec={"Utilización (%)": 1})
        st.caption("Nodo → cliente")
        mostrar_tabla(rnc[["Nodo", "Cliente", "Distancia (km)", "Cantidad", "Capacidad", "Utilización (%)"]]
                      .sort_values("Utilización (%)", ascending=False), dec={"Utilización (%)": 1})
        grafico(gr.red_logistica(parametros, r["flujo_pn"], r["flujo_nc"], alto=470))
        st.caption("Grosor de cada ruta proporcional a su uso. Las rutas punteadas no se utilizan en el plan óptimo.")

    with t3:
        nc = r["flujo_nc"].groupby(["Cliente", "id_nodo", "Nodo"], as_index=False)["Cantidad"].sum()
        fig = go.Figure()
        for n_id, g in nc.groupby("id_nodo"):
            fig.add_bar(y=g["Cliente"], x=g["Cantidad"], name=g["Nodo"].iloc[0], orientation="h",
                        marker_color=gr.COLOR_NODO.get(n_id, gr.SERIE[0]),
                        hovertemplate="%{y}: %{x:,.0f} u<extra>" + g["Nodo"].iloc[0] + "</extra>")
        fig.update_layout(barmode="stack")
        fig.update_xaxes(title="Unidades", showgrid=True, gridcolor=gr.REJILLA)
        fig.update_yaxes(title=None)
        gr.barras_redondeadas(fig)
        grafico(gr.layout(fig, "¿Cuánto recibe cada cliente y por cuál nodo?", 360))
        mostrar_tabla(r["clientes"].drop(columns=["ID_CLIENTE"]), dec={"Nivel de servicio (%)": 1})
        st.markdown("##### Entregas por producto y cliente (unidades)")
        ent = r["flujo_nc"].pivot_table(index="Producto", columns="Cliente", values="Cantidad", aggfunc="sum")
        ent["Total"] = ent.sum(axis=1)
        mostrar_tabla(ent.reset_index())
        st.markdown("##### Detalle de trasbordo y distribución")
        a, b = st.columns(2)
        with a:
            st.caption("Envíos planta → nodo")
            fpn = r["flujo_pn"][r["flujo_pn"]["Cantidad"] > 0.5][["Planta", "Nodo", "Producto", "Cantidad", "Costo unitario"]]
            mostrar_tabla(fpn, alto=360)
            descargar(r["flujo_pn"], "plan_trasbordo.csv", "Descargar trasbordo")
        with b:
            st.caption("Envíos nodo → cliente")
            fnc = r["flujo_nc"][r["flujo_nc"]["Cantidad"] > 0.5][["Nodo", "Cliente", "Producto", "Cantidad", "Costo unitario"]]
            mostrar_tabla(fnc, alto=360)
            descargar(r["flujo_nc"], "plan_distribucion.csv", "Descargar distribución")

    with t4:
        st.markdown("Las restricciones activas son las capacidades que se usan al 100 % y por tanto limitan el plan. "
                    "El **precio sombra** indica cuánto bajaría el costo total (COP) por cada unidad adicional de "
                    "esa capacidad.")
        act = r["activas"].sort_values("Precio sombra (COP/u)", ascending=False)
        mostrar_tabla(act)
        lim = act[act["Precio sombra (COP/u)"] > 0].head(3)
        if len(lim):
            st.markdown("**Cuellos de botella con mayor efecto en el costo:** " + "; ".join(
                f"{x['Restricción']} {x['Elemento']} ({pesos(x['Precio sombra (COP/u)'])} por unidad)"
                for _, x in lim.iterrows()) + ".")
        st.markdown("##### Capacidad agregada frente a la demanda")
        mostrar_tabla(r["diagnostico"], dec={"Uso requerido (%)": 1})

    with t5:
        st.markdown("**Conjuntos:** $P$ productos (10), $C$ clientes (5), $F$ plantas (4), $N$ nodos de trasbordo (2).")
        st.markdown("**Parámetros:** $D_{pc}$ demanda pronosticada · $Cap_{fp}$ capacidad de la planta por producto · "
                    "$CapMax_f$ capacidad total de la planta · $cp_p$ costo unitario de producción · "
                    "$t_{fnp}$ costo planta → nodo · $d_{ncp}$ costo nodo → cliente · $CapN_n$ capacidad del nodo · "
                    "$U_{fn}$ y $V_{nc}$ capacidades de transporte. Todos en unidades por mes y COP por unidad.")
        st.markdown("**Variables:** $x_{fp}$ producción · $y_{fnp}$ envío planta → nodo · $z_{ncp}$ envío nodo → cliente"
                    + (" · $s_{pc}$ demanda no atendida" if permitir_faltantes else "") + ".")
        st.latex(r"\min Z=\sum_{f,p} cp_p\,x_{fp}+\sum_{f,n,p} t_{fnp}\,y_{fnp}+\sum_{n,c,p} d_{ncp}\,z_{ncp}"
                 + (r"+\sum_{p,c} \pi_p\, s_{pc}" if permitir_faltantes else ""))
        st.latex(r"\sum_{n} z_{ncp}" + (r"+s_{pc}" if permitir_faltantes else "") + r"=D_{pc}\quad\forall p,c")
        st.latex(r"x_{fp}\le Cap_{fp}\qquad \sum_p x_{fp}\le CapMax_f")
        st.latex(r"x_{fp}=\sum_n y_{fnp}\qquad \sum_f y_{fnp}=\sum_c z_{ncp}")
        st.latex(r"\sum_{f,p} y_{fnp}\le CapN_n\qquad \sum_p y_{fnp}\le U_{fn}\qquad \sum_p z_{ncp}\le V_{nc}")
        st.latex(r"x,\;y,\;z" + (r",\;s" if permitir_faltantes else "") + r"\ \ge 0")
        st.caption(f"El costo de producción es el costo base por producto de Datos.xlsx. La demanda pronosticada se "
                   f"redondea a unidades enteras. Resuelto con CBC a través de PuLP · estado: {r['estado']}.")


# ===========================================================================
# 6. ESCENARIOS
# ===========================================================================
def tarjeta_escenario(col, nombre, r, base_costo):
    desc = esc.ESCENARIOS.get(nombre, {}).get("descripcion") or f"Demanda {r['ajuste']:+.0%} y capacidades ajustadas"
    est = esc._estado(r)
    clase = {"Óptimo": "ok", "Óptimo con faltantes": "alerta", "Infactible": "error"}[est]
    if r["factible"]:
        delta = ""
        if base_costo and nombre != "Base":
            cambio = (r["costo_total"] / base_costo - 1) * 100
            delta = f" · {'+' if cambio >= 0 else '−'}{miles(abs(cambio), 1)} % frente a la base"
        servicio = (1 - r["faltante_total"] / max(r["demanda_total"], 1)) * 100
        cuerpo = (f'<div class="cifra">{pesos(r["costo_total"], millones=True)}</div>'
                  f'<div class="linea">Costo total{delta}</div>'
                  f'<div class="linea">Demanda: {miles(r["demanda_total"])} u · Servicio: {miles(servicio, 1)} %</div>')
    else:
        aux = r.get("diagnostico_faltantes") or {}
        falt = aux.get("faltante_total")
        cuerpo = (f'<div class="cifra">Sin plan factible</div>'
                  f'<div class="linea">Demanda: {miles(r["demanda_total"])} u</div>'
                  + (f'<div class="linea">Faltarían al menos {miles(falt)} u</div>' if falt else ""))
    col.markdown(f'<div class="tarjeta-esc" style="border-top:4px solid {gr.COLOR_ESCENARIO.get(nombre, gr.SERIE[0])}">'
                 f'<div class="nom">{nombre} <span class="etq {clase}">{est}</span></div>'
                 f'<div class="desc">{desc}</div>{cuerpo}</div>', unsafe_allow_html=True)


def barras_agrupadas(df, x, y, titulo, ytitulo, alto=380, linea_100=False, texto_pct=False):
    fig = go.Figure()
    for nombre in df["Escenario"].unique():
        g = df[df["Escenario"] == nombre]
        fig.add_bar(x=g[x], y=g[y], name=nombre, marker_color=gr.COLOR_ESCENARIO.get(nombre, gr.SERIE[0]),
                    text=[f"{miles(v, 1)} %" if texto_pct else miles(v) for v in g[y]],
                    textposition="outside", textfont=dict(size=10, color=gr.TINTA_2),
                    hovertemplate="%{x}<br>" + nombre + ": %{y:,.1f}<extra></extra>")
    fig.update_layout(barmode="group", bargroupgap=0.08)
    fig.update_yaxes(title=ytitulo, range=[0, max(df[y].max() * 1.18, 110 if linea_100 else 0)])
    fig.update_xaxes(title=None)
    if linea_100:
        fig.add_hline(y=100, line_dash="dash", line_color=gr.CRITICO, line_width=1.5,
                      annotation_text="100 %", annotation_position="top left", annotation_font_color=gr.CRITICO)
    gr.barras_redondeadas(fig)
    return gr.layout(fig, titulo, alto)


def pagina_escenarios():
    encabezado("Análisis de escenarios",
               "Efecto de los cambios en la demanda sobre costos, uso de la capacidad y atención a los clientes.")
    with st.expander("Agregar un escenario personalizado", expanded=False):
        usar = st.toggle("Incluir escenario personalizado", value=False)
        a, b, c, d = st.columns(4)
        aj = a.slider("Cambio en la demanda (%)", -40, 60, 25, 5, disabled=not usar)
        cp = b.slider("Capacidad de plantas (%)", 50, 150, 100, 5, disabled=not usar)
        cn = c.slider("Capacidad de nodos (%)", 50, 150, 100, 5, disabled=not usar)
        ct = d.slider("Capacidad de rutas (%)", 50, 150, 100, 5, disabled=not usar)
        st.caption("Sirve para probar, por ejemplo, un crecimiento fuerte de la demanda o el cierre parcial de una planta.")

    res = dict(resultados)
    if usar:
        res["Personalizado"] = correr_escenario("Personalizado", demanda_base, clave_param, parametros, aj / 100,
                                                permitir_faltantes, factor_penal, cp / 100, cn / 100, ct / 100)
    base_costo = res["Base"]["costo_total"] if res["Base"]["factible"] else None

    cols = st.columns(len(res))
    for col, (nombre, r) in zip(cols, res.items()):
        tarjeta_escenario(col, nombre, r, base_costo)
    st.write("")

    costos = esc.comparar_costos(res)
    plantas = esc.comparar_plantas(res)
    nodos = esc.comparar_nodos(res)
    clientes = esc.comparar_clientes(res)
    restr = esc.comparar_restricciones(res)

    t1, t2, t3, t4, t5, t6 = st.tabs(["Costos", "Plantas", "Nodos de trasbordo", "Clientes",
                                      "Restricciones y factibilidad", "Conclusiones"])
    factibles = costos[costos["Estado"] != "Infactible"]
    with t1:
        if len(factibles):
            a, b = st.columns(2)
            with a:
                fig = go.Figure(go.Bar(
                    x=factibles["Escenario"], y=factibles["Total operación"] / 1e6,
                    marker_color=[gr.COLOR_ESCENARIO.get(e, gr.SERIE[0]) for e in factibles["Escenario"]],
                    text=[f"$ {miles(v / 1e6, 1)} M" for v in factibles["Total operación"]], textposition="outside",
                    hovertemplate="%{x}: $ %{y:,.1f} M<extra></extra>"))
                fig.update_yaxes(title="Millones de COP", range=[0, factibles["Total operación"].max() / 1e6 * 1.18])
                gr.barras_redondeadas(fig)
                grafico(gr.layout(fig, "Costo total de producción y distribución", 380))
            with b:
                comp = factibles.melt(id_vars="Escenario", value_vars=["Transporte planta-nodo", "Distribución nodo-cliente"],
                                      var_name="Componente", value_name="Costo")
                comp["Costo"] = comp["Costo"] / 1e6
                fig = go.Figure()
                for e in comp["Escenario"].unique():
                    g = comp[comp["Escenario"] == e]
                    fig.add_bar(x=g["Componente"], y=g["Costo"], name=e,
                                marker_color=gr.COLOR_ESCENARIO.get(e, gr.SERIE[0]),
                                text=[f"{miles(v, 1)}" for v in g["Costo"]], textposition="outside",
                                hovertemplate="%{x}<br>" + e + ": $ %{y:,.2f} M<extra></extra>")
                fig.update_layout(barmode="group")
                fig.update_yaxes(title="Millones de COP", range=[0, comp["Costo"].max() * 1.2])
                gr.barras_redondeadas(fig)
                grafico(gr.layout(fig, "Costos logísticos (transporte y distribución)", 380))
        cols_dec = {k: 0 for k in costos.columns}
        mostrar_tabla(costos, dec=cols_dec)
        st.caption("Costos en COP. La penalización por faltantes no se suma al costo de operación; "
                   "se muestra aparte cuando se permiten faltantes.")
    with t2:
        if len(plantas):
            grafico(barras_agrupadas(plantas, "Planta", "Producción (u)", "Cantidades producidas por planta",
                                     "Unidades"))
            grafico(barras_agrupadas(plantas, "Planta", "Utilización (%)", "Utilización de la capacidad de las plantas",
                                     "Utilización (%)", linea_100=True, texto_pct=True))
            mostrar_tabla(plantas, dec={"Utilización (%)": 1})
    with t3:
        if len(nodos):
            grafico(barras_agrupadas(nodos, "Nodo", "Flujo (u)", "Unidades que pasan por cada nodo", "Unidades"))
            grafico(barras_agrupadas(nodos, "Nodo", "Utilización (%)", "Utilización de los nodos de trasbordo",
                                     "Utilización (%)", linea_100=True, texto_pct=True))
            mostrar_tabla(nodos, dec={"Utilización (%)": 1})
    with t4:
        if len(clientes):
            grafico(barras_agrupadas(clientes, "Cliente", "Entregado (u)", "Cantidades distribuidas a cada cliente",
                                     "Unidades", alto=400))
            mostrar_tabla(clientes, dec={"Nivel de servicio (%)": 1})
    with t5:
        for nombre, r in res.items():
            if not r["factible"]:
                st.markdown(f"#### Escenario {nombre}")
                mostrar_infactible(r)
            elif r["faltante_total"] > 0.5:
                st.markdown(f"#### Escenario {nombre}")
                estado(f"Se dejan de atender {miles(r['faltante_total'])} unidades por falta de capacidad.", "alerta")
                f = r["atencion"]
                mostrar_tabla(f[f["Faltante"] > 0.5][["Producto", "Cliente", "Demanda", "Entregado", "Faltante"]])
        if len(restr):
            st.markdown("##### Capacidades al 100 % en cada escenario")
            filtro = st.multiselect("Escenarios", list(restr["Escenario"].unique()),
                                    default=list(restr["Escenario"].unique()))
            mostrar_tabla(restr[restr["Escenario"].isin(filtro)], alto=420)
            st.caption("Mientras más restricciones se saturan, menor es el margen de la red para absorber "
                       "aumentos de demanda; un precio sombra alto señala dónde ampliar capacidad tiene más valor.")
        st.markdown("##### Capacidad agregada frente a la demanda")
        diag = pd.concat([r["diagnostico"].assign(Escenario=n) for n, r in res.items()], ignore_index=True)
        diag = diag[["Escenario", "Recurso", "Demanda", "Capacidad", "Holgura", "Uso requerido (%)", "Estado"]]
        mostrar_tabla(diag, dec={"Uso requerido (%)": 1}, alto=420,
                      estilo=lambda s: s.apply(lambda r: ["color:#b42318;font-weight:600" if r["Estado"] == "Insuficiente"
                                                          else "" for _ in r], axis=1))
    with t6:
        for linea in conclusiones(res):
            st.markdown(f"- {linea}")


def conclusiones(res):
    out = []
    b = res["Base"]
    if not b["factible"]:
        return ["El escenario base no es factible: la capacidad de la red no alcanza para la demanda pronosticada."]
    for nombre in [n for n in res if n != "Base"]:
        r = res[nombre]
        if r["factible"]:
            dc = (r["costo_total"] / b["costo_total"] - 1) * 100
            dd = (r["demanda_total"] / b["demanda_total"] - 1) * 100
            out.append(f"**{nombre}:** la demanda cambia {miles(dd, 1)} % y el costo total {miles(dc, 1)} % "
                       f"(diferencia de {pesos(r['costo_total'] - b['costo_total'], millones=True)}).")
            sat_p = r["plantas"][r["plantas"]["Utilización (%)"] >= 99.5]["Planta"].tolist()
            sat_n = r["nodos"][r["nodos"]["Utilización (%)"] >= 99.5]["Nodo"].tolist()
            if sat_p or sat_n:
                out.append(f"En el escenario {nombre} trabajan al 100 %: " + ", ".join(sat_p + sat_n) + ".")
            if r["faltante_total"] > 0.5:
                out.append(f"En el escenario {nombre} quedan {miles(r['faltante_total'])} unidades sin atender.")
        else:
            aux = r.get("diagnostico_faltantes") or {}
            txt = f"**{nombre}:** el modelo es infactible; la red no puede atender toda la demanda"
            if aux.get("factible"):
                txt += f" (faltarían al menos {miles(aux['faltante_total'])} unidades)"
            insuf = r["diagnostico"][r["diagnostico"]["Estado"] == "Insuficiente"]["Recurso"].tolist()
            if insuf:
                txt += ". Recursos insuficientes: " + ", ".join(insuf)
            out.append(txt + ".")
    if b["factible"]:
        holg = b["plantas"].assign(H=lambda d: d["Capacidad"] - d["Cantidad"]).sort_values("H", ascending=False).iloc[0]
        out.append(f"En el escenario base la planta con mayor capacidad libre es {holg['Planta']} "
                   f"({miles(holg['Capacidad'] - holg['Cantidad'])} u, {miles(100 - holg['Utilización (%)'], 1)} % disponible).")
        rutas = b["activas"][b["activas"]["Restricción"].str.contains("ruta")]["Elemento"].tolist()
        if rutas:
            out.append("Rutas que ya operan a su límite en el escenario base: " + ", ".join(rutas)
                       + ". Ampliarlas permitiría usar más la capacidad libre de las plantas.")
        nodo_max = b["nodos"].sort_values("Utilización (%)", ascending=False).iloc[0]
        out.append(f"El nodo más exigido es {nodo_max['Nodo']} con {miles(nodo_max['Utilización (%)'], 1)} % de su "
                   "capacidad de manipulación.")
    return out


# ===========================================================================
# 7. PARÁMETROS DE LA RED
# ===========================================================================
def pagina_parametros():
    encabezado("Parámetros de la red logística", "Información leída de Datos.xlsx: unidades por mes y COP por unidad.")
    t1, t2, t3, t4 = st.tabs(["Catálogos", "Capacidades", "Costos de transporte", "Otras hojas"])
    with t1:
        st.markdown("##### Productos")
        mostrar_tabla(parametros["productos"])
        a, b = st.columns(2)
        with a:
            st.markdown("##### Plantas")
            mostrar_tabla(parametros["plantas"])
            st.markdown("##### Nodos de trasbordo")
            mostrar_tabla(parametros["nodos"])
        with b:
            st.markdown("##### Clientes")
            mostrar_tabla(parametros["clientes"])
    with t2:
        cpp = parametros["cap_planta_producto"].merge(
            parametros["productos"][["id_producto", "nombre"]], on="id_producto")
        piv = cpp.pivot_table(index="nombre", columns="id_planta", values="capacidad_mensual")
        piv = piv.reindex(columns=[p for p in parametros["plantas"]["id_planta"] if p in piv.columns])
        texto = piv.map(lambda v: "" if pd.isna(v) else miles(v))
        fig = go.Figure(go.Heatmap(z=piv.values, x=piv.columns, y=piv.index, colorscale=gr.AZUL_SEC, text=texto.values,
                                   texttemplate="%{text}", hoverongaps=False, colorbar=dict(title="u/mes"),
                                   hovertemplate="%{y} en %{x}: %{z:,.0f} u/mes<extra></extra>"))
        fig.update_xaxes(side="top", showgrid=False)
        fig.update_yaxes(showgrid=False, autorange="reversed")
        grafico(gr.layout(fig, "Capacidad mensual por planta y producto (vacío: no lo fabrica)", 450, margin=dict(t=90)))
        a, b = st.columns(2)
        with a:
            st.markdown("##### Rutas planta → nodo")
            mostrar_tabla(parametros["rutas_pn"])
        with b:
            st.markdown("##### Rutas nodo → cliente")
            mostrar_tabla(parametros["rutas_nc"])
    with t3:
        nombres = dict(zip(parametros["productos"]["id_producto"], parametros["productos"]["nombre"]))
        pn = parametros["costo_planta_nodo"].assign(Ruta=lambda d: d["id_planta"] + " → " + d["id_nodo"])
        piv = pn.pivot_table(index="id_producto", columns="Ruta", values="costo_unitario").rename(index=nombres)
        st.markdown("##### Costo unitario planta → nodo (COP/u)")
        mostrar_tabla(piv.reset_index().rename(columns={"id_producto": "Producto"}))
        nc = parametros["costo_nodo_cliente"].assign(Ruta=lambda d: d["id_nodo"] + " → " + d["id_cliente"])
        piv = nc.pivot_table(index="id_producto", columns="Ruta", values="costo_unitario").rename(index=nombres)
        st.markdown("##### Costo unitario nodo → cliente (COP/u)")
        mostrar_tabla(piv.reset_index().rename(columns={"id_producto": "Producto"}))
    with t4:
        st.caption("Hojas de consulta. El modelo es de un solo período y envía todo a través de los nodos, por lo que "
                   "no usa el transporte directo planta → cliente ni los costos de inventario.")
        for nombre, df in parametros["extra"].items():
            st.markdown(f"##### {nombre.replace('_', ' ').capitalize()}")
            mostrar_tabla(df, alto=300)


PAGINAS = {
    "Resumen general": pagina_resumen,
    "Datos y ETL": pagina_etl,
    "Análisis descriptivo": pagina_descriptiva,
    "Pronósticos": pagina_pronosticos,
    "Optimización": pagina_optimizacion,
    "Escenarios": pagina_escenarios,
    "Parámetros de la red": pagina_parametros,
}
PAGINAS[seccion]()
