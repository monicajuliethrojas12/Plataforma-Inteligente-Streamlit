"""
Carga de información de la plataforma.

- Lee el libro Datos.xlsx (productos, clientes, plantas, nodos, capacidades y costos).
- Lee la demanda histórica si viene en el libro o en un archivo aparte (CSV / Excel).
- Si no se entrega la demanda histórica, la reconstruye con el mismo procedimiento
  de la Fase 1 (semilla 42), que es la base usada en el informe.
"""

import io
import unicodedata

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Catálogo de la Fase 1 (mismo orden y parámetros del cuaderno del proyecto)
# ---------------------------------------------------------------------------
CATALOGO = pd.DataFrame(
    [
        ("P01", "Cuadernos escolares", "ESTACIONAL", 48000, 1.40, 2400),
        ("P02", "Juguetes didácticos", "ESTACIONAL", 85000, 1.50, 1200),
        ("P03", "Chocolatería de regalo", "ESTACIONAL", 62000, 1.45, 1800),
        ("P04", "Ventiladores de torre", "ESTACIONAL", 140000, 1.35, 900),
        ("P05", "Impermeables industriales", "ESTACIONAL", 95000, 1.40, 1100),
        ("P06", "Paneles solares portátiles", "TENDENCIA", 210000, 1.45, 700),
        ("P07", "Baterías de litio recargables", "TENDENCIA", 175000, 1.50, 800),
        ("P08", "Envases plásticos tradicionales", "TENDENCIA", 38000, 1.30, 3000),
        ("P09", "Bolsas biodegradables", "TENDENCIA", 42000, 1.40, 2200),
        ("P10", "Bombillos incandescentes", "TENDENCIA", 52000, 1.30, 1600),
    ],
    columns=["id", "nombre", "categoria", "costo", "margen", "base"],
)

CLIENTES_BASE = pd.DataFrame(
    [
        ("C1", "Comercial Caribe S.A.S.", "Barranquilla"),
        ("C2", "Grupo Andino Retail", "Bogotá"),
        ("C3", "Distribuciones Antioqueñas", "Medellín"),
        ("C4", "Almacenes del Pacífico", "Cali"),
        ("C5", "Mayoristas del Eje Cafetero", "Pereira"),
    ],
    columns=["id", "nombre", "ciudad"],
)

PLANTAS_BASE = {
    "BOG": ("Planta Bogotá", "Bogotá"),
    "MED": ("Planta Medellín", "Medellín"),
    "CAL": ("Planta Cali", "Cali"),
    "BGA": ("Planta Bucaramanga", "Bucaramanga"),
}

# Nombres alternos que aparecen entre el libro de parámetros y la base histórica
ALIAS_PRODUCTO = {
    "caja de chocolates": "P03",
    "chocolateria de regalo": "P03",
    "chocolates": "P03",
}

HOJAS_REQUERIDAS = [
    "Producto",
    "Cliente",
    "Planta",
    "Nodo",
    "capacidad_planta_producto",
    "costo_transporte_planta_nodo",
    "costo_distribucion_nodo_cliente",
]


def normalizar(texto):
    """Minúsculas, sin tildes y sin espacios extra (para comparar nombres)."""
    if texto is None or (isinstance(texto, float) and np.isnan(texto)):
        return ""
    texto = unicodedata.normalize("NFKD", str(texto))
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return " ".join(texto.lower().split())


# ---------------------------------------------------------------------------
# Base histórica de la Fase 1
# ---------------------------------------------------------------------------
def generar_base_historica(seed=42, n_meses=36, fecha_inicio="2023-11-01"):
    """Reproduce la base sintética de demanda (1.800 registros) de la Fase 1."""
    rng = np.random.default_rng(seed)
    productos = CATALOGO.copy()
    productos["precio_base"] = (productos.costo * productos.margen).round(-2)
    clientes = CLIENTES_BASE.copy()
    plantas = PLANTAS_BASE

    idx = {
        "P01": [1.9, 1.7, 0.9, 0.7, 0.6, 0.7, 1.0, 1.6, 1.2, 0.7, 0.6, 0.7],
        "P02": [0.6, 0.7, 0.6, 0.8, 0.8, 0.7, 0.8, 0.7, 0.9, 1.3, 1.9, 2.4],
        "P03": [0.7, 1.4, 0.7, 0.7, 1.5, 0.8, 0.7, 0.8, 1.4, 0.9, 1.0, 2.0],
        "P04": [1.4, 1.6, 1.8, 1.5, 1.0, 0.7, 0.6, 0.6, 0.7, 0.7, 0.7, 0.7],
        "P05": [0.6, 0.5, 0.9, 1.5, 1.6, 0.8, 0.5, 0.5, 0.9, 1.6, 1.7, 0.9],
    }
    idx = {k: np.array(v) / np.mean(v) for k, v in idx.items()}
    tend = {"P06": 0.020, "P07": 0.015, "P08": -0.014, "P09": 0.012, "P10": -0.018}
    desc = {"C1": 0.00, "C2": -0.02, "C3": -0.03, "C4": 0.01, "C5": 0.00}
    pesos_cli = np.array([0.20, 0.30, 0.22, 0.16, 0.12])
    eleg = {
        "BOG": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "MED": [1, 1, 1, 0, 1, 0, 1, 1, 1, 1],
        "CAL": [1, 0, 1, 1, 0, 1, 1, 1, 0, 1],
        "BGA": [0, 1, 1, 1, 1, 1, 0, 1, 1, 0],
    }
    d = {
        ("Bogotá", "Medellín"): 415, ("Bogotá", "Cali"): 460, ("Bogotá", "Bucaramanga"): 390,
        ("Bogotá", "Pereira"): 300, ("Bogotá", "Barranquilla"): 1000, ("Medellín", "Cali"): 420,
        ("Medellín", "Bucaramanga"): 590, ("Medellín", "Barranquilla"): 680,
        ("Medellín", "Pereira"): 220, ("Cali", "Bucaramanga"): 870,
        ("Cali", "Barranquilla"): 1130, ("Cali", "Pereira"): 190,
        ("Bucaramanga", "Barranquilla"): 570, ("Bucaramanga", "Pereira"): 640,
    }

    def dist(a, b):
        return 15 if a == b else (d.get((a, b)) or d[(b, a)])

    fechas = pd.date_range(fecha_inicio, periods=n_meses, freq="MS")
    tc = (n_meses - 1) / 2
    filas = []
    for k, p in productos.iterrows():
        w = rng.dirichlet(pesos_cli * 60)
        sd = 0.06 if p.categoria == "ESTACIONAL" else 0.07
        elegibles = [f for f in plantas if eleg[f][k]]
        for j, c in clientes.iterrows():
            planta = min(elegibles, key=lambda f: dist(plantas[f][1], c.ciudad))
            for t, f in enumerate(fechas):
                if p.categoria == "ESTACIONAL":
                    nivel = (1 + 0.002 * (t - tc)) * idx[p.id][f.month - 1]
                else:
                    nivel = max(0.15, 1 + tend[p.id] * (t - tc))
                dem = max(0, int(round(p.base * w[j] * nivel * (1 + rng.normal(0, sd)))))
                precio = round(
                    p.precio_base * (1 + desc[c.id]) * 1.004 ** t * (1 + rng.normal(0, 0.005)), -1
                )
                filas.append((f, p.nombre, c.nombre, dem, plantas[planta][0], precio))

    plana = (
        pd.DataFrame(filas, columns=["Fecha", "Producto", "Cliente", "Demanda", "Planta", "Precio Unitario"])
        .sort_values(["Fecha", "Producto", "Cliente"])
        .reset_index(drop=True)
    )
    plana["Precio Unitario"] = plana["Precio Unitario"].astype(int)
    return plana


# ---------------------------------------------------------------------------
# Lectura del libro de parámetros
# ---------------------------------------------------------------------------
def _leer_bytes(archivo):
    if archivo is None:
        return None
    if isinstance(archivo, (bytes, bytearray)):
        return bytes(archivo)
    if hasattr(archivo, "getvalue"):
        return archivo.getvalue()
    with open(archivo, "rb") as fh:
        return fh.read()


def _limpiar_columnas(df):
    df = df.copy()
    df.columns = [normalizar(c).replace(" ", "_") for c in df.columns]
    return df


def _hoja(libro, nombre):
    """Busca una hoja sin importar mayúsculas, tildes o espacios."""
    objetivo = normalizar(nombre)
    for k in libro:
        if normalizar(k) == objetivo:
            return libro[k]
    raise KeyError(f"No se encontró la hoja '{nombre}' en el archivo.")


def _a_numero(serie):
    return pd.to_numeric(serie, errors="coerce")


def leer_parametros(archivo):
    """Lee Datos.xlsx y devuelve un diccionario con todos los parámetros de la red."""
    contenido = _leer_bytes(archivo)
    libro = pd.read_excel(io.BytesIO(contenido), sheet_name=None)

    faltan = [h for h in HOJAS_REQUERIDAS if normalizar(h) not in {normalizar(k) for k in libro}]
    if faltan:
        raise ValueError("Al archivo le faltan las hojas: " + ", ".join(faltan))

    # Productos
    prod = _limpiar_columnas(_hoja(libro, "Producto"))
    prod = prod[prod["id_producto"].astype(str).str.match(r"^P\d+")].copy()
    prod["id_producto"] = prod["id_producto"].astype(str).str.strip()
    prod["nombre"] = prod["nombre"].astype(str).str.strip()
    prod["costo_base_cop"] = _a_numero(prod["costo_base_cop"])
    prod["precio_base_cop"] = _a_numero(prod["precio_base_cop"])
    prod = prod.rename(columns={"peso_[kg]": "peso_kg"})
    prod = prod.reset_index(drop=True)

    # Clientes
    cli = _limpiar_columnas(_hoja(libro, "Cliente"))
    cli = cli[cli["id_cliente"].astype(str).str.match(r"^C\d+")].copy()
    cli["id_cliente"] = cli["id_cliente"].astype(str).str.strip()
    cli["nombre"] = cli["nombre"].astype(str).str.strip()
    cli = cli.reset_index(drop=True)

    # Plantas (la hoja trae un ejemplo debajo de la tabla; solo se toman los códigos válidos)
    pla = _limpiar_columnas(_hoja(libro, "Planta"))
    pla["capacidad_max_mensual"] = _a_numero(pla["capacidad_max_mensual"])
    pla = pla[pla["capacidad_max_mensual"].notna() & pla["ciudad"].notna()].copy()
    pla["id_planta"] = pla["id_planta"].astype(str).str.strip()
    pla["nombre"] = pla["nombre"].astype(str).str.strip()
    pla = pla.reset_index(drop=True)

    # Nodos
    nod = _limpiar_columnas(_hoja(libro, "Nodo"))
    nod["capacidad_manipulacion_mensual"] = _a_numero(nod["capacidad_manipulacion_mensual"])
    nod = nod[nod["capacidad_manipulacion_mensual"].notna()].copy()
    nod["id_nodo"] = nod["id_nodo"].astype(str).str.strip()
    nod["nombre"] = nod["nombre"].astype(str).str.strip()
    nod = nod.reset_index(drop=True)

    # Capacidad planta-producto
    cpp = _limpiar_columnas(_hoja(libro, "capacidad_planta_producto"))
    cpp = cpp[["id_planta", "id_producto", "capacidad_mensual"]].copy()
    cpp["capacidad_mensual"] = _a_numero(cpp["capacidad_mensual"])
    cpp = cpp.dropna().reset_index(drop=True)
    for c in ["id_planta", "id_producto"]:
        cpp[c] = cpp[c].astype(str).str.strip()

    # Transporte planta -> nodo. La capacidad viene una vez por planta y aplica a cada ruta.
    tpn = _limpiar_columnas(_hoja(libro, "costo_transporte_planta_nodo"))
    for c in ["distancia_km", "capacidad_transporte", "costo_unitario"]:
        tpn[c] = _a_numero(tpn[c])
    tpn = tpn.dropna(subset=["id_planta", "id_nodo", "id_producto", "costo_unitario"]).copy()
    for c in ["id_planta", "id_nodo", "id_producto"]:
        tpn[c] = tpn[c].astype(str).str.strip()
    cap_ruta = tpn.groupby(["id_planta", "id_nodo"])["capacidad_transporte"].max()
    cap_planta = tpn.groupby("id_planta")["capacidad_transporte"].max()
    rutas_pn = (
        tpn.groupby(["id_planta", "id_nodo"], as_index=False)["distancia_km"].first()
    )
    rutas_pn["capacidad_transporte"] = [
        cap_ruta.get((f, n)) if pd.notna(cap_ruta.get((f, n))) else cap_planta.get(f)
        for f, n in zip(rutas_pn["id_planta"], rutas_pn["id_nodo"])
    ]

    # Distribución nodo -> cliente
    tnc = _limpiar_columnas(_hoja(libro, "costo_distribucion_nodo_cliente"))
    for c in ["distancia_km", "capacidad_transporte", "costo_unitario"]:
        tnc[c] = _a_numero(tnc[c])
    tnc = tnc.dropna(subset=["id_nodo", "id_cliente", "id_producto", "costo_unitario"]).copy()
    for c in ["id_nodo", "id_cliente", "id_producto"]:
        tnc[c] = tnc[c].astype(str).str.strip()
    rutas_nc = tnc.groupby(["id_nodo", "id_cliente"], as_index=False).agg(
        distancia_km=("distancia_km", "first"), capacidad_transporte=("capacidad_transporte", "max")
    )

    # Hojas complementarias (solo de consulta)
    extra = {}
    for nombre in ["costo_transporte_planta_cliente", "costo_inventario_planta", "costo_inventario_nodo"]:
        try:
            extra[nombre] = _limpiar_columnas(_hoja(libro, nombre)).dropna(how="all")
        except KeyError:
            pass

    return {
        "productos": prod,
        "clientes": cli,
        "plantas": pla,
        "nodos": nod,
        "cap_planta_producto": cpp,
        "costo_planta_nodo": tpn[["id_planta", "id_nodo", "id_producto", "distancia_km", "costo_unitario"]],
        "costo_nodo_cliente": tnc[["id_nodo", "id_cliente", "id_producto", "distancia_km", "costo_unitario"]],
        "rutas_pn": rutas_pn,
        "rutas_nc": rutas_nc,
        "extra": extra,
        "libro": libro,
    }


# ---------------------------------------------------------------------------
# Demanda histórica
# ---------------------------------------------------------------------------
COLUMNAS_DEMANDA = {"fecha", "producto", "cliente", "demanda"}


def _estandarizar_demanda(df):
    mapa = {}
    for c in df.columns:
        n = normalizar(c)
        if n in ("fecha", "periodo"):
            mapa[c] = "Fecha"
        elif n in ("producto", "nombre_producto"):
            mapa[c] = "Producto"
        elif n in ("cliente", "nombre_cliente"):
            mapa[c] = "Cliente"
        elif n in ("demanda", "demanda_total", "unidades"):
            mapa[c] = "Demanda"
        elif n == "planta":
            mapa[c] = "Planta"
        elif n in ("precio unitario", "precio_unitario", "precio", "precio_promedio"):
            mapa[c] = "Precio Unitario"
    df = df.rename(columns=mapa)
    df = df.loc[:, ~df.columns.duplicated()]
    return df


def demanda_desde_libro(archivo):
    """Busca en el libro una hoja con Fecha, Producto, Cliente y Demanda."""
    contenido = _leer_bytes(archivo)
    libro = pd.read_excel(io.BytesIO(contenido), sheet_name=None)
    for nombre, hoja in libro.items():
        cols = {normalizar(c) for c in hoja.columns}
        cols |= {normalizar(c).replace(" ", "_") for c in hoja.columns}
        if {"fecha", "producto", "cliente"} <= cols and ({"demanda"} & cols or "demanda_total" in cols):
            return _estandarizar_demanda(hoja), nombre
    return None, None


def leer_demanda(archivo, nombre_archivo=""):
    """Lee la demanda histórica desde CSV o Excel."""
    contenido = _leer_bytes(archivo)
    nombre = (nombre_archivo or getattr(archivo, "name", "") or "").lower()
    if nombre.endswith(".csv"):
        try:
            df = pd.read_csv(io.BytesIO(contenido), encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(io.BytesIO(contenido), encoding="latin-1")
        if df.shape[1] == 1:  # separador punto y coma
            df = pd.read_csv(io.BytesIO(contenido), sep=";", encoding="utf-8-sig")
        df = _estandarizar_demanda(df)
    else:
        df, _ = demanda_desde_libro(contenido)
        if df is None:
            raise ValueError("El archivo no tiene una hoja con columnas Fecha, Producto, Cliente y Demanda.")
    faltan = {"Fecha", "Producto", "Cliente", "Demanda"} - set(df.columns)
    if faltan:
        raise ValueError("Faltan columnas en la demanda: " + ", ".join(sorted(faltan)))
    return df


# ---------------------------------------------------------------------------
# Correspondencia de nombres entre la demanda y los parámetros
# ---------------------------------------------------------------------------
def mapa_productos(parametros):
    """Diccionario nombre normalizado -> id de producto."""
    mapa = dict(ALIAS_PRODUCTO)
    for _, r in CATALOGO.iterrows():
        mapa[normalizar(r["nombre"])] = r["id"]
    for _, r in parametros["productos"].iterrows():
        mapa[normalizar(r["nombre"])] = r["id_producto"]
        mapa[normalizar(r["id_producto"])] = r["id_producto"]
    return mapa


def mapa_clientes(parametros):
    mapa = {}
    for _, r in parametros["clientes"].iterrows():
        mapa[normalizar(r["nombre"])] = r["id_cliente"]
        mapa[normalizar(r["id_cliente"])] = r["id_cliente"]
    return mapa


def asignar_ids(df, parametros):
    """Agrega ID_PRODUCTO e ID_CLIENTE a una tabla con columnas Producto y Cliente."""
    mp, mc = mapa_productos(parametros), mapa_clientes(parametros)
    df = df.copy()
    df["ID_PRODUCTO"] = df["Producto"].map(lambda x: mp.get(normalizar(x)))
    df["ID_CLIENTE"] = df["Cliente"].map(lambda x: mc.get(normalizar(x)))
    return df
