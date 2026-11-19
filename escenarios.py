"""
Fase 6. Análisis de escenarios.

Escenarios definidos en el proyecto:
  - Base:       demanda pronosticada por los modelos seleccionados.
  - Optimista:  demanda 10 % inferior (menor exigencia sobre la red).
  - Pesimista:  demanda 10 % superior (mayor exigencia sobre la red).
Se puede agregar un escenario personalizado con otro ajuste de demanda y de capacidades.

Para cada escenario se comparan: costo total, producción por planta, utilización de
plantas y nodos, cantidades entregadas a cada cliente y las restricciones que limitan
(o impiden) atender toda la demanda.
"""

import numpy as np
import pandas as pd

import optimizacion as opt

ESCENARIOS = {
    "Base": {"ajuste": 0.00, "descripcion": "Demanda pronosticada por los modelos"},
    "Optimista": {"ajuste": -0.10, "descripcion": "Demanda 10 % inferior"},
    "Pesimista": {"ajuste": 0.10, "descripcion": "Demanda 10 % superior"},
}


def demanda_escenario(demanda_base, ajuste):
    """Aplica el ajuste porcentual a la demanda pronosticada (unidades enteras)."""
    d = demanda_base.copy()
    d["Demanda"] = np.round(d["Pronóstico"] * (1 + ajuste)).clip(lower=0).astype(int)
    return d


def ejecutar_escenario(nombre, demanda_base, parametros, ajuste, permitir_faltantes=False,
                       factor_penalizacion=2.0, factor_cap_plantas=1.0, factor_cap_nodos=1.0,
                       factor_cap_transporte=1.0):
    d = demanda_escenario(demanda_base, ajuste)
    caps = dict(factor_cap_plantas=factor_cap_plantas, factor_cap_nodos=factor_cap_nodos,
                factor_cap_transporte=factor_cap_transporte)
    res = opt.resolver(d, parametros, permitir_faltantes=permitir_faltantes,
                       factor_penalizacion=factor_penalizacion, **caps)
    res["nombre"] = nombre
    res["ajuste"] = ajuste
    res["demanda"] = d
    res["diagnostico"] = opt.diagnostico_capacidad(d, parametros, **caps)

    # Si exige atender toda la demanda y no es posible, se informa la infactibilidad
    # y se calcula aparte cuánto quedaría sin atender (solo como diagnóstico).
    if not res["factible"]:
        aux = opt.resolver(d, parametros, permitir_faltantes=True,
                           factor_penalizacion=max(factor_penalizacion, 2.0), **caps)
        res["diagnostico_faltantes"] = aux
    return res


def ejecutar_todos(demanda_base, parametros, escenarios=None, **opciones):
    escenarios = escenarios or {k: v["ajuste"] for k, v in ESCENARIOS.items()}
    return {nombre: ejecutar_escenario(nombre, demanda_base, parametros, ajuste, **opciones)
            for nombre, ajuste in escenarios.items()}


# ---------------------------------------------------------------------------
# Tablas comparativas
# ---------------------------------------------------------------------------
def comparar_costos(resultados):
    filas = []
    for nombre, r in resultados.items():
        fila = {"Escenario": nombre, "Ajuste demanda": f"{r['ajuste']:+.0%}",
                "Demanda (u)": r["demanda_total"], "Estado": _estado(r)}
        if r["factible"]:
            fila.update(r["costos"])
            fila["Costo por unidad entregada"] = r["costo_total"] / max(r["demanda_total"] - r["faltante_total"], 1)
            fila["Demanda no atendida (u)"] = r["faltante_total"]
        filas.append(fila)
    return pd.DataFrame(filas)


def comparar_plantas(resultados):
    filas = []
    for nombre, r in resultados.items():
        if not r["factible"]:
            continue
        for _, p in r["plantas"].iterrows():
            filas.append({"Escenario": nombre, "Planta": p["Planta"], "Producción (u)": p["Cantidad"],
                          "Capacidad (u)": p["Capacidad"], "Utilización (%)": p["Utilización (%)"]})
    return pd.DataFrame(filas)


def comparar_nodos(resultados):
    filas = []
    for nombre, r in resultados.items():
        if not r["factible"]:
            continue
        for _, n in r["nodos"].iterrows():
            filas.append({"Escenario": nombre, "Nodo": n["Nodo"], "Flujo (u)": n["Cantidad"],
                          "Capacidad (u)": n["Capacidad"], "Utilización (%)": n["Utilización (%)"]})
    return pd.DataFrame(filas)


def comparar_clientes(resultados):
    filas = []
    for nombre, r in resultados.items():
        if not r["factible"]:
            continue
        for _, c in r["clientes"].iterrows():
            filas.append({"Escenario": nombre, "Cliente": c["Cliente"], "Demanda (u)": c["Demanda"],
                          "Entregado (u)": c["Entregado"], "Faltante (u)": c["Faltante"],
                          "Nivel de servicio (%)": c["Nivel de servicio (%)"]})
    return pd.DataFrame(filas)


def comparar_restricciones(resultados):
    """Restricciones de capacidad activas (al 100 %) en cada escenario."""
    filas = []
    for nombre, r in resultados.items():
        fuente = r if r["factible"] else r.get("diagnostico_faltantes", {})
        if not fuente or not fuente.get("factible"):
            continue
        act = fuente["activas"].copy()
        act.insert(0, "Escenario", nombre)
        filas.append(act)
    return pd.concat(filas, ignore_index=True) if filas else pd.DataFrame()


def _estado(r):
    if r["factible"]:
        if r.get("faltante_total", 0) > 0.5:
            return "Óptimo con faltantes"
        return "Óptimo"
    return "Infactible"
