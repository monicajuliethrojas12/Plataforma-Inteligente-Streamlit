"""Estilo visual común y gráficos reutilizables del tablero."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

# Paleta categórica (orden fijo) y colores de interfaz
SERIE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
TENUE = "#898781"
REJILLA = "#e1e0d9"
EJE = "#c3c2b7"
SUPERFICIE = "#ffffff"
AZUL_SEC = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
OK, ALERTA, CRITICO = "#0ca30c", "#fab219", "#d03b3b"

COLOR_MODELO = {"Holt-Winters": SERIE[0], "MLP": SERIE[1], "Holt": SERIE[2], "Promedio móvil": SERIE[3]}
COLOR_ESCENARIO = {"Base": SERIE[0], "Optimista": SERIE[2], "Pesimista": SERIE[1], "Personalizado": SERIE[6]}
COLOR_PLANTA = {"BOG": SERIE[0], "MED": SERIE[1], "CAL": SERIE[2], "BGA": SERIE[3]}
COLOR_NODO = {"NN": SERIE[4], "NC": SERIE[6]}
COLOR_TIPO = {"planta": "#1c5cab", "nodo": "#6250d6", "cliente": "#1baf7a"}

FUENTE = "Inter, 'Segoe UI', system-ui, -apple-system, sans-serif"

pio.templates["plataforma"] = go.layout.Template(
    layout=dict(
        font=dict(family=FUENTE, size=13, color=TINTA_2),
        title=dict(font=dict(size=15, color=TINTA), x=0.005, xanchor="left", xref="container",
                   y=0.985, yanchor="top", yref="container"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=SERIE,
        margin=dict(l=10, r=10, t=78, b=10),
        xaxis=dict(showgrid=False, linecolor=EJE, ticks="outside", tickcolor=EJE, zeroline=False, automargin=True,
                   tickfont=dict(color=TINTA_2), title=dict(font=dict(size=12, color=TENUE))),
        yaxis=dict(gridcolor=REJILLA, gridwidth=1, linecolor=EJE, zeroline=False, rangemode="tozero", automargin=True,
                   tickfont=dict(color=TINTA_2), title=dict(font=dict(size=12, color=TENUE))),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(size=12, color=TINTA_2), bgcolor="rgba(0,0,0,0)", title_text=""),
        hoverlabel=dict(bgcolor="white", bordercolor=REJILLA, font=dict(family=FUENTE, size=12, color=TINTA)),
        bargap=0.25,
        separators=",.",
    )
)
pio.templates.default = "plataforma"


def miles(v, dec=0):
    """Formato colombiano: 1.234.567,8"""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    txt = f"{v:,.{dec}f}"
    return txt.replace(",", "§").replace(".", ",").replace("§", ".")


def pesos(v, millones=False):
    if millones:
        return f"$ {miles(v / 1e6, 1)} M"
    return f"$ {miles(v)}"


def partir(texto):
    """Parte un nombre largo en dos líneas para etiquetas de ejes."""
    palabras = str(texto).split()
    if len(palabras) < 2 or len(str(texto)) < 14:
        return str(texto)
    mitad = len(str(texto)) / 2
    acum, corte = 0, 1
    for i, p in enumerate(palabras[:-1], start=1):
        acum += len(p) + 1
        if acum >= mitad:
            corte = i
            break
    return " ".join(palabras[:corte]) + "<br>" + " ".join(palabras[corte:])


MESES_CORTO =["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]


def eje_meses(fig, fechas, cada=6):
    """Etiquetas de fecha en español sobre el eje x."""
    vals = sorted(pd.to_datetime(pd.Series(list(fechas))).dropna().unique())
    vals = [pd.Timestamp(v) for v in vals][::cada]
    fig.update_xaxes(tickvals=vals, ticktext=[f"{MESES_CORTO[v.month - 1]}<br>{v.year}" for v in vals],
                     hoverformat="%m/%Y")
    return fig


def layout(fig, titulo=None, alto=380, **kw):
    fig.update_layout(title=titulo, height=alto, **kw)
    return fig


def barras_redondeadas(fig):
    try:
        fig.update_traces(marker_cornerradius=4, selector=dict(type="bar"))
    except ValueError:
        pass
    return fig


# ---------------------------------------------------------------------------
# Diagrama de la red logística
# ---------------------------------------------------------------------------
def red_logistica(parametros, flujo_pn=None, flujo_nc=None, alto=520):
    """Plantas -> nodos -> clientes. Sin flujos muestra capacidades; con flujos muestra cantidades."""
    plantas = parametros["plantas"]
    nodos = parametros["nodos"]
    clientes = parametros["clientes"]
    rutas_pn = parametros["rutas_pn"]
    rutas_nc = parametros["rutas_nc"]

    pos = {}
    for i, f in enumerate(plantas["id_planta"]):
        pos[f] = (0.0, 1 - (i + 0.5) / len(plantas))
    for i, n in enumerate(nodos["id_nodo"]):
        pos[n] = (0.5, 1 - (i + 0.5) / len(nodos))
    for i, c in enumerate(clientes["id_cliente"]):
        pos[c] = (1.0, 1 - (i + 0.5) / len(clientes))

    fig = go.Figure()
    usar_flujo = flujo_pn is not None
    if usar_flujo:
        q_pn = flujo_pn.groupby(["id_planta", "id_nodo"])["Cantidad"].sum()
        q_nc = flujo_nc.groupby(["id_nodo", "id_cliente"])["Cantidad"].sum()

    def arco(a, b, cap, km, q=None, color="#9ec5f4"):
        (x0, y0), (x1, y1) = pos[a], pos[b]
        activo = q is None or q > 0.5
        ancho = 1.5 if q is None else (1 + 7 * min(q / max(cap, 1), 1) if activo else 1)
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1], mode="lines",
            line=dict(color=color if activo else "#dcdad3", width=ancho, dash=None if activo else "dot"),
            hoverinfo="text",
            text=(f"{a} → {b}<br>Distancia: {miles(km)} km<br>Capacidad: {miles(cap)} u/mes"
                  + ("" if q is None else f"<br>Flujo óptimo: {miles(q)} u ({miles(q / cap * 100, 1)} %)")),
            showlegend=False,
        ))

    for _, r in rutas_pn.iterrows():
        q = None if not usar_flujo else float(q_pn.get((r.id_planta, r.id_nodo), 0))
        arco(r.id_planta, r.id_nodo, r.capacidad_transporte, r.distancia_km, q, "#6da7ec")
    for _, r in rutas_nc.iterrows():
        q = None if not usar_flujo else float(q_nc.get((r.id_nodo, r.id_cliente), 0))
        arco(r.id_nodo, r.id_cliente, r.capacidad_transporte, r.distancia_km, q, "#b3a9f0")

    def nodos_tipo(df, idcol, tipo, etiqueta, cap=None):
        xs = [pos[k][0] for k in df[idcol]]
        ys = [pos[k][1] for k in df[idcol]]
        textos = [etiqueta(r) for _, r in df.iterrows()]
        hover = [f"<b>{r['nombre']}</b> ({r['ciudad']})" + (f"<br>Capacidad: {miles(r[cap])} u/mes" if cap else "")
                 for _, r in df.iterrows()]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers+text", text=textos,
            textposition="middle left" if tipo == "planta" else (
                "middle right" if tipo == "cliente" else
                ["top center" if i == 0 else "bottom center" for i in range(len(xs))]),
            textfont=dict(size=12, color=TINTA),
            marker=dict(size=26 if tipo == "nodo" else 20, color=COLOR_TIPO[tipo], line=dict(color="white", width=2)),
            hovertext=hover, hoverinfo="text", showlegend=False,
        ))

    nodos_tipo(plantas, "id_planta", "planta",
               lambda r: f"{r['nombre'].replace('Planta ', '')}<br>{miles(r['capacidad_max_mensual'])} u/mes  ",
               "capacidad_max_mensual")
    nodos_tipo(nodos, "id_nodo", "nodo",
               lambda r: f"{r['nombre']} · {r['ciudad']}<br>{miles(r['capacidad_manipulacion_mensual'])} u/mes",
               "capacidad_manipulacion_mensual")
    nodos_tipo(clientes, "id_cliente", "cliente",
               lambda r: f"  {r['nombre']}<br>  {r['ciudad']}")

    for x, t in [(0, "PLANTAS"), (0.5, "NODOS DE TRASBORDO"), (1, "CLIENTES")]:
        fig.add_annotation(x=x, y=1.07, text=t, showarrow=False, font=dict(size=11, color=TENUE))
    fig.update_xaxes(visible=False, range=[-0.38, 1.62])
    fig.update_yaxes(visible=False, range=[-0.05, 1.1])
    fig.update_layout(height=alto, margin=dict(l=0, r=0, t=20, b=0), hovermode="closest")
    return fig


def sankey(flujo_pn, flujo_nc, alto=480):
    pn = flujo_pn.groupby(["Planta", "id_planta", "Nodo"], as_index=False)["Cantidad"].sum()
    nc = flujo_nc.groupby(["Nodo", "Cliente"], as_index=False)["Cantidad"].sum()
    pn, nc = pn[pn["Cantidad"] > 0.5], nc[nc["Cantidad"] > 0.5]
    etiquetas = list(dict.fromkeys(list(pn["Planta"]) + list(pn["Nodo"]) + list(nc["Cliente"])))
    idx = {e: i for i, e in enumerate(etiquetas)}
    colores_nodo = []
    for e in etiquetas:
        if e in set(pn["Planta"]):
            colores_nodo.append(COLOR_TIPO["planta"])
        elif e in set(pn["Nodo"]):
            colores_nodo.append(COLOR_TIPO["nodo"])
        else:
            colores_nodo.append(COLOR_TIPO["cliente"])

    def rgba(h, a):
        h = h.lstrip("#")
        return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"

    fuentes = [idx[a] for a in pn["Planta"]] + [idx[a] for a in nc["Nodo"]]
    destinos = [idx[b] for b in pn["Nodo"]] + [idx[b] for b in nc["Cliente"]]
    valores = list(pn["Cantidad"]) + list(nc["Cantidad"])
    col_links = [rgba(COLOR_PLANTA.get(f, SERIE[0]), 0.35) for f in pn["id_planta"]] + \
                [rgba("#6250d6", 0.22)] * len(nc)
    fig = go.Figure(go.Sankey(
        arrangement="snap",
        valueformat=",.0f",
        valuesuffix=" u",
        node=dict(label=etiquetas, color=colores_nodo, pad=22, thickness=16,
                  line=dict(color="white", width=1)),
        link=dict(source=fuentes, target=destinos, value=valores, color=col_links),
    ))
    fig.update_layout(height=alto, margin=dict(l=10, r=10, t=10, b=10), font=dict(size=12, color=TINTA))
    return fig


def barras_utilizacion(df, etiqueta, valor="Utilización (%)", color_col=None, colores=None, alto=320, titulo=None):
    """Barras horizontales de utilización con referencia al 100 %."""
    colores_barra = []
    for _, r in df.iterrows():
        if colores and color_col:
            colores_barra.append(colores.get(r[color_col], SERIE[0]))
        else:
            colores_barra.append(SERIE[0])
    fig = go.Figure(go.Bar(
        x=df[valor], y=df[etiqueta], orientation="h", marker_color=colores_barra,
        text=[f"{miles(v, 1)} %" for v in df[valor]], textposition="outside",
        textfont=dict(color=TINTA_2),
        hovertemplate="%{y}: %{x:.1f} %<extra></extra>",
    ))
    fig.add_vline(x=100, line_dash="dash", line_color=CRITICO, line_width=1.5,
                  annotation_text="100 %", annotation_position="top", annotation_font_color=CRITICO)
    fig.update_xaxes(range=[0, max(115, float(df[valor].max()) * 1.15)], title="Utilización (%)",
                     showgrid=True, gridcolor=REJILLA)
    fig.update_yaxes(autorange="reversed", showgrid=False, title=None)
    barras_redondeadas(fig)
    return layout(fig, titulo, alto, showlegend=False)
