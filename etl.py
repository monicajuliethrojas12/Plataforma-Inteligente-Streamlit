"""
Fase 2. Proceso ETL y analítica descriptiva.

Mismo tratamiento del cuaderno del proyecto:
limpieza de texto, tipos, duplicados, faltantes (mediana por producto y luego global),
registros inválidos, atípicos por IQR y consolidación por producto, cliente y período.
"""

import numpy as np
import pandas as pd


MESES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]


def mes_texto(fecha, largo=False):
    nombres = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
               "septiembre", "octubre", "noviembre", "diciembre"]
    fecha = pd.Timestamp(fecha)
    return f"{(nombres if largo else MESES)[fecha.month - 1]} {fecha.year}"


def detectar_atipicos(datos):
    q1, q3 = datos.quantile(0.25), datos.quantile(0.75)
    iqr = q3 - q1
    return (datos < q1 - 1.5 * iqr) | (datos > q3 + 1.5 * iqr)


def ejecutar_etl(plana):
    df = plana.copy(deep=True)
    originales = len(df)

    if "Planta" not in df.columns:
        df["Planta"] = "Sin asignar"
    if "Precio Unitario" not in df.columns:
        df["Precio Unitario"] = np.nan

    # Texto
    for col in ["Producto", "Cliente", "Planta"]:
        df[col] = df[col].astype("string").str.strip().replace("", pd.NA)

    # Tipos
    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce", dayfirst=False)
    df["Demanda"] = pd.to_numeric(df["Demanda"], errors="coerce")
    df["Precio Unitario"] = pd.to_numeric(df["Precio Unitario"], errors="coerce")

    # Duplicados exactos
    duplicados = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()

    # Faltantes
    faltantes_por_col = df.isna().sum()
    faltantes = int(faltantes_por_col.sum())
    for col in ["Demanda", "Precio Unitario"]:
        mediana_prod = df.groupby("Producto")[col].transform("median")
        df[col] = df[col].fillna(mediana_prod).fillna(df[col].median())
    df["Demanda"] = df["Demanda"].round()

    # Inconsistencias
    invalidos = (
        df["Fecha"].isna() | df["Producto"].isna() | df["Cliente"].isna()
        | df["Demanda"].isna() | (df["Demanda"] < 0)
    )
    if df["Precio Unitario"].notna().any():
        invalidos |= df["Precio Unitario"].isna() | (df["Precio Unitario"] <= 0)
    eliminados = int(invalidos.sum())
    df = df.loc[~invalidos].copy()
    df["Demanda"] = df["Demanda"].astype(int)

    # Fechas: todas al primer día del mes
    df["Fecha"] = df["Fecha"].dt.to_period("M").dt.to_timestamp()

    # Atípicos (se marcan, no se eliminan)
    df["Atipico_Demanda"] = df.groupby("Producto")["Demanda"].transform(detectar_atipicos)
    df["Atipico_Precio"] = df.groupby("Producto")["Precio Unitario"].transform(detectar_atipicos)

    base = df.sort_values(["Fecha", "Producto", "Cliente"]).reset_index(drop=True)

    resumen = pd.DataFrame(
        {
            "Indicador": [
                "Registros originales",
                "Registros duplicados eliminados",
                "Datos faltantes iniciales",
                "Registros eliminados por inconsistencias",
                "Atípicos en demanda (IQR, conservados)",
                "Atípicos en precio (IQR, conservados)",
                "Registros finales limpios",
            ],
            "Resultado": [
                originales, duplicados, faltantes, eliminados,
                int(base["Atipico_Demanda"].sum()), int(base["Atipico_Precio"].sum()), len(base),
            ],
        }
    )

    # Consistencia de fechas y cantidades
    periodos = pd.period_range(base["Fecha"].min(), base["Fecha"].max(), freq="M")
    conteo = base.groupby(["Producto", "Cliente"])["Fecha"].nunique()
    dup_clave = int(base.duplicated(["Fecha", "Producto", "Cliente"]).sum())
    validacion = pd.DataFrame(
        {
            "Verificación": [
                "Período cubierto",
                "Meses en el horizonte",
                "Productos",
                "Clientes",
                "Series producto-cliente",
                "Series con todos los meses",
                "Registros repetidos (fecha, producto, cliente)",
                "Demandas negativas",
            ],
            "Resultado": [
                f"{mes_texto(base['Fecha'].min())} a {mes_texto(base['Fecha'].max())}",
                len(periodos),
                base["Producto"].nunique(),
                base["Cliente"].nunique(),
                len(conteo),
                f"{int((conteo == len(periodos)).sum())} de {len(conteo)}",
                dup_clave,
                int((base["Demanda"] < 0).sum()),
            ],
        }
    )
    validacion["Resultado"] = validacion["Resultado"].astype(str)
    return base, resumen, validacion, faltantes_por_col


def consolidar(base):
    """Demanda consolidada por producto, cliente y período (insumo de los modelos)."""
    tabla = (
        base.groupby(["Producto", "Cliente", "Fecha"], as_index=False)
        .agg(Demanda=("Demanda", "sum"), Precio_Promedio=("Precio Unitario", "mean"))
        .sort_values(["Producto", "Cliente", "Fecha"])
        .reset_index(drop=True)
    )
    tabla["Precio_Promedio"] = tabla["Precio_Promedio"].round(2)
    return tabla


def base_modelos(consolidada):
    """Base con variables temporales, cíclicas y rezagos (1, 2, 3 y media de 3)."""
    df = consolidada.copy()
    df["Anio"] = df["Fecha"].dt.year
    df["Mes"] = df["Fecha"].dt.month
    df["Mes_sin"] = np.sin(2 * np.pi * df["Mes"] / 12).round(4)
    df["Mes_cos"] = np.cos(2 * np.pi * df["Mes"] / 12).round(4)
    g = df.groupby(["Producto", "Cliente"])["Demanda"]
    for k in (1, 2, 3):
        df[f"Lag_{k}"] = g.shift(k)
    df["Media_3"] = g.transform(lambda s: s.shift(1).rolling(3).mean()).round(2)
    return df


def variabilidad(df, por):
    """Estadísticos de la demanda mensual consolidada por la dimensión indicada."""
    mensual = df.groupby([por, "Fecha"], as_index=False)["Demanda"].sum()
    est = mensual.groupby(por)["Demanda"].agg(["mean", "std", "min", "max"]).reset_index()
    est.columns = [por, "Promedio", "Desv. estándar", "Mínimo", "Máximo"]
    est["CV (%)"] = est["Desv. estándar"] / est["Promedio"] * 100
    return est.sort_values("CV (%)", ascending=False).reset_index(drop=True)
