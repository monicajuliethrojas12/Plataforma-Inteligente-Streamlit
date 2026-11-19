"""
Fase 5. Modelo de programación lineal de costo mínimo (PuLP).

Conjuntos:  P productos, C clientes, F plantas, N nodos de trasbordo.
Variables:  x[f,p]   producción del producto p en la planta f
            y[f,n,p] envío de p desde la planta f al nodo n
            z[n,c,p] envío de p desde el nodo n al cliente c
            s[p,c]   demanda no atendida (solo si se permiten faltantes)
Objetivo:   min  Σ cp·x + Σ t·y + Σ d·z (+ Σ penalización·s)
Restricciones:
  1. Satisfacción de la demanda        Σn z[n,c,p] (+ s[p,c]) = D[p,c]
  2. Capacidad de producción           x[f,p] ≤ Cap[f,p] ;  Σp x[f,p] ≤ CapMax[f]
  3. Balance en plantas                x[f,p] = Σn y[f,n,p]
  4. Balance en nodos                  Σf y[f,n,p] = Σc z[n,c,p]
  5. Capacidad de los nodos            Σf,p y[f,n,p] ≤ CapNodo[n]
  6. Capacidad de transporte           Σp y[f,n,p] ≤ U[f,n] ;  Σp z[n,c,p] ≤ V[n,c]
  7. No negatividad
"""

import numpy as np
import pandas as pd
import pulp

TOL = 1e-6


def _agregar(modelo, restriccion, nombre):
    """Agrega la restricción y devuelve su nombre (para consultar holgura y precio sombra)."""
    modelo.addConstraint(restriccion, nombre)
    return nombre


def _dic(df, claves, valor):
    return {tuple(r[k] for k in claves) if len(claves) > 1 else r[claves[0]]: float(r[valor])
            for _, r in df.iterrows()}


def preparar_demanda(pronostico, parametros):
    """Agrupa el pronóstico por (producto, cliente) en unidades enteras."""
    from datos import asignar_ids

    d = asignar_ids(pronostico, parametros)
    sin_id = d[d["ID_PRODUCTO"].isna() | d["ID_CLIENTE"].isna()]
    if len(sin_id):
        raise ValueError(
            "No se pudo asociar con Datos.xlsx: "
            + ", ".join(sorted(set(sin_id["Producto"].astype(str)) | set(sin_id["Cliente"].astype(str))))
        )
    d = d.groupby(["ID_PRODUCTO", "ID_CLIENTE", "Producto", "Cliente"], as_index=False)["Pronóstico"].sum()
    d["Demanda"] = np.round(d["Pronóstico"]).astype(int)
    return d


def resolver(demanda, parametros, permitir_faltantes=False, factor_penalizacion=2.0,
             factor_cap_plantas=1.0, factor_cap_nodos=1.0, factor_cap_transporte=1.0):
    """Resuelve el modelo. `demanda` debe traer ID_PRODUCTO, ID_CLIENTE y Demanda."""
    prod = parametros["productos"]
    P = list(prod["id_producto"])
    C = list(parametros["clientes"]["id_cliente"])
    F = list(parametros["plantas"]["id_planta"])
    N = list(parametros["nodos"]["id_nodo"])

    D = {(r.ID_PRODUCTO, r.ID_CLIENTE): float(r.Demanda) for r in demanda.itertuples()}
    costo_prod = _dic(prod, ["id_producto"], "costo_base_cop")
    precio = _dic(prod, ["id_producto"], "precio_base_cop")
    cap_fp = {k: v * factor_cap_plantas for k, v in
              _dic(parametros["cap_planta_producto"], ["id_planta", "id_producto"], "capacidad_mensual").items()}
    cap_f = {k: v * factor_cap_plantas for k, v in
             _dic(parametros["plantas"], ["id_planta"], "capacidad_max_mensual").items()}
    cap_n = {k: v * factor_cap_nodos for k, v in
             _dic(parametros["nodos"], ["id_nodo"], "capacidad_manipulacion_mensual").items()}
    U = {k: v * factor_cap_transporte for k, v in
         _dic(parametros["rutas_pn"], ["id_planta", "id_nodo"], "capacidad_transporte").items()}
    V = {k: v * factor_cap_transporte for k, v in
         _dic(parametros["rutas_nc"], ["id_nodo", "id_cliente"], "capacidad_transporte").items()}
    t = _dic(parametros["costo_planta_nodo"], ["id_planta", "id_nodo", "id_producto"], "costo_unitario")
    d = _dic(parametros["costo_nodo_cliente"], ["id_nodo", "id_cliente", "id_producto"], "costo_unitario")

    FP = [k for k in cap_fp if k[0] in F and k[1] in P]
    FNP = [(f, n, p) for (f, p) in FP for n in N if (f, n) in U and (f, n, p) in t]
    NCP = [(n, c, p) for n in N for c in C for p in P if (n, c) in V and (n, c, p) in d]

    m = pulp.LpProblem("plan_produccion_distribucion", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", FP, lowBound=0)
    y = pulp.LpVariable.dicts("y", FNP, lowBound=0)
    z = pulp.LpVariable.dicts("z", NCP, lowBound=0)
    s = pulp.LpVariable.dicts("s", list(D.keys()), lowBound=0) if permitir_faltantes else {}
    penal = {p: factor_penalizacion * precio[p] for p in P}

    c_prod = pulp.lpSum(costo_prod[p] * x[(f, p)] for (f, p) in FP)
    c_tr = pulp.lpSum(t[k] * y[k] for k in FNP)
    c_dis = pulp.lpSum(d[k] * z[k] for k in NCP)
    c_pen = pulp.lpSum(penal[p] * s[(p, c)] for (p, c) in s) if s else 0
    m += c_prod + c_tr + c_dis + c_pen

    R = {}  # (tipo, elemento) -> restricción
    for (p, c), dem in D.items():
        entra = pulp.lpSum(z[(n, c, p)] for n in N if (n, c, p) in z)
        R[("Demanda", f"{p}-{c}")] = _agregar(m, 
            (entra + (s[(p, c)] if s else 0)) == dem, f"dem_{p}_{c}")
    for (f, p) in FP:
        R[("Cap. planta-producto", f"{f}-{p}")] = _agregar(m, x[(f, p)] <= cap_fp[(f, p)], f"cfp_{f}_{p}")
        m.addConstraint(x[(f, p)] == pulp.lpSum(y[(f, n, p)] for n in N if (f, n, p) in y), f"bal_f_{f}_{p}")
    for f in F:
        R[("Cap. planta", f)] = _agregar(m, 
            pulp.lpSum(x[(ff, p)] for (ff, p) in FP if ff == f) <= cap_f[f], f"cf_{f}")
    for n in N:
        for p in P:
            m.addConstraint(
                pulp.lpSum(y[(f, n, p)] for f in F if (f, n, p) in y)
                == pulp.lpSum(z[(n, c, p)] for c in C if (n, c, p) in z), f"bal_n_{n}_{p}")
        R[("Cap. nodo", n)] = _agregar(m, 
            pulp.lpSum(y[k] for k in FNP if k[1] == n) <= cap_n[n], f"cn_{n}")
    for (f, n) in U:
        if f in F and n in N:
            R[("Cap. ruta planta-nodo", f"{f}-{n}")] = _agregar(m, 
                pulp.lpSum(y[k] for k in FNP if k[0] == f and k[1] == n) <= U[(f, n)], f"u_{f}_{n}")
    for (n, c) in V:
        if n in N and c in C:
            R[("Cap. ruta nodo-cliente", f"{n}-{c}")] = _agregar(m, 
                pulp.lpSum(z[k] for k in NCP if k[0] == n and k[1] == c) <= V[(n, c)], f"v_{n}_{c}")

    m.solve(pulp.PULP_CBC_CMD(msg=False))
    estado = pulp.LpStatus[m.status]

    res = {
        "estado": estado,
        "factible": estado == "Optimal",
        "permitir_faltantes": permitir_faltantes,
        "demanda_total": sum(D.values()),
        "capacidades": {"plantas": cap_f, "nodos": cap_n},
    }
    if estado != "Optimal":
        return res

    val = lambda v: max(v.value() or 0.0, 0.0)
    # Nombres tal como aparecen en la demanda (para que coincidan con los pronósticos)
    nombres_p = dict(zip(prod["id_producto"], prod["nombre"]))
    if "Producto" in demanda.columns:
        nombres_p.update(dict(zip(demanda["ID_PRODUCTO"], demanda["Producto"])))
    nombres_c = dict(zip(parametros["clientes"]["id_cliente"], parametros["clientes"]["nombre"]))
    nombres_f = dict(zip(parametros["plantas"]["id_planta"], parametros["plantas"]["nombre"]))
    nombres_n = dict(zip(parametros["nodos"]["id_nodo"], parametros["nodos"]["nombre"]))

    produccion = pd.DataFrame(
        [{"id_planta": f, "id_producto": p, "Planta": nombres_f[f], "Producto": nombres_p[p],
          "Cantidad": val(x[(f, p)]), "Capacidad": cap_fp[(f, p)]} for (f, p) in FP])
    produccion["Utilización (%)"] = produccion["Cantidad"] / produccion["Capacidad"] * 100

    flujo_pn = pd.DataFrame(
        [{"id_planta": f, "id_nodo": n, "id_producto": p, "Planta": nombres_f[f], "Nodo": nombres_n[n],
          "Producto": nombres_p[p], "Cantidad": val(y[(f, n, p)]), "Costo unitario": t[(f, n, p)]}
         for (f, n, p) in FNP])
    flujo_nc = pd.DataFrame(
        [{"id_nodo": n, "id_cliente": c, "id_producto": p, "Nodo": nombres_n[n], "Cliente": nombres_c[c],
          "Producto": nombres_p[p], "Cantidad": val(z[(n, c, p)]), "Costo unitario": d[(n, c, p)]}
         for (n, c, p) in NCP])

    plantas = produccion.groupby(["id_planta", "Planta"], as_index=False)["Cantidad"].sum()
    plantas["Capacidad"] = plantas["id_planta"].map(cap_f)
    plantas["Utilización (%)"] = plantas["Cantidad"] / plantas["Capacidad"] * 100
    plantas = plantas.set_index("id_planta").reindex(F).reset_index()

    nodos = flujo_pn.groupby(["id_nodo", "Nodo"], as_index=False)["Cantidad"].sum()
    nodos["Capacidad"] = nodos["id_nodo"].map(cap_n)
    nodos["Utilización (%)"] = nodos["Cantidad"] / nodos["Capacidad"] * 100
    nodos = nodos.set_index("id_nodo").reindex(N).reset_index()

    faltante = {k: (val(s[k]) if s else 0.0) for k in D}
    atencion = demanda[["ID_PRODUCTO", "ID_CLIENTE", "Producto", "Cliente", "Demanda"]].copy()
    atencion["Faltante"] = [faltante[(p, c)] for p, c in zip(atencion["ID_PRODUCTO"], atencion["ID_CLIENTE"])]
    atencion["Entregado"] = atencion["Demanda"] - atencion["Faltante"]
    clientes = atencion.groupby(["ID_CLIENTE", "Cliente"], as_index=False)[["Demanda", "Entregado", "Faltante"]].sum()
    clientes["Nivel de servicio (%)"] = np.where(
        clientes["Demanda"] > 0, clientes["Entregado"] / clientes["Demanda"] * 100, 100.0)

    costos = {
        "Producción": pulp.value(c_prod) or 0.0,
        "Transporte planta-nodo": pulp.value(c_tr) or 0.0,
        "Distribución nodo-cliente": pulp.value(c_dis) or 0.0,
        "Penalización por faltantes": (pulp.value(c_pen) or 0.0) if s else 0.0,
    }
    costos["Total operación"] = costos["Producción"] + costos["Transporte planta-nodo"] + costos["Distribución nodo-cliente"]

    # Restricciones activas (holgura nula) y su precio sombra
    activas = []
    for (tipo, elem), nombre in R.items():
        if tipo == "Demanda":
            continue
        r = m.constraints[nombre]
        holgura = r.slack if r.slack is not None else 0.0
        limite = -r.constant
        usado = limite - holgura
        if limite > 0 and holgura <= 1e-4 * max(1.0, limite):
            activas.append({"Restricción": tipo, "Elemento": elem, "Capacidad": limite,
                            "Usado": usado, "Precio sombra (COP/u)": abs(r.pi or 0.0)})
    activas = pd.DataFrame(activas, columns=["Restricción", "Elemento", "Capacidad", "Usado", "Precio sombra (COP/u)"])

    res.update({
        "costo_total": costos["Total operación"],
        "objetivo": pulp.value(m.objective),
        "costos": costos,
        "produccion": produccion,
        "flujo_pn": flujo_pn,
        "flujo_nc": flujo_nc,
        "plantas": plantas,
        "nodos": nodos,
        "clientes": clientes,
        "atencion": atencion,
        "faltante_total": float(sum(faltante.values())),
        "activas": activas,
    })
    return res


def diagnostico_capacidad(demanda, parametros, factor_cap_plantas=1.0, factor_cap_nodos=1.0,
                          factor_cap_transporte=1.0):
    """Revisión rápida de capacidad agregada contra la demanda (explica infactibilidades)."""
    filas = []
    prod = parametros["productos"].set_index("id_producto")["nombre"].to_dict()
    if "Producto" in demanda.columns:
        prod.update(dict(zip(demanda["ID_PRODUCTO"], demanda["Producto"])))
    cap_p = parametros["cap_planta_producto"].groupby("id_producto")["capacidad_mensual"].sum() * factor_cap_plantas
    dem_p = demanda.groupby("ID_PRODUCTO")["Demanda"].sum()
    for p, dem in dem_p.items():
        filas.append({"Recurso": f"Producción de {prod.get(p, p)}", "Demanda": dem, "Capacidad": cap_p.get(p, 0)})
    filas.append({"Recurso": "Producción total (4 plantas)", "Demanda": dem_p.sum(),
                  "Capacidad": parametros["plantas"]["capacidad_max_mensual"].sum() * factor_cap_plantas})
    filas.append({"Recurso": "Manipulación en nodos", "Demanda": dem_p.sum(),
                  "Capacidad": parametros["nodos"]["capacidad_manipulacion_mensual"].sum() * factor_cap_nodos})
    cli = parametros["clientes"].set_index("id_cliente")["nombre"]
    cap_c = parametros["rutas_nc"].groupby("id_cliente")["capacidad_transporte"].sum() * factor_cap_transporte
    for c, dem in demanda.groupby("ID_CLIENTE")["Demanda"].sum().items():
        filas.append({"Recurso": f"Rutas de llegada a {cli.get(c, c)}", "Demanda": dem, "Capacidad": cap_c.get(c, 0)})
    out = pd.DataFrame(filas)
    out["Holgura"] = out["Capacidad"] - out["Demanda"]
    out["Uso requerido (%)"] = out["Demanda"] / out["Capacidad"] * 100
    out["Estado"] = np.where(out["Holgura"] < 0, "Insuficiente", "Suficiente")
    return out
