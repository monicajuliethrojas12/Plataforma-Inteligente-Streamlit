"""
Fases 3 y 4. Modelos de pronóstico, evaluación y selección automática.

Cuatro métodos aplicados a cada serie producto-cliente:
  - Promedio móvil simple (ventana de 3 meses)
  - Holt (nivel + tendencia)
  - Holt-Winters (tendencia aditiva, estacionalidad multiplicativa, ciclo de 12 meses)
  - Red neuronal MLP (6 rezagos + mes en seno/coseno + índice de tiempo; 32 y 16 neuronas, ReLU, Adam)

Validación walk-forward: los últimos 6 meses se pronostican uno a uno usando solo
la información real disponible hasta el mes anterior. Con las tres métricas
(MAE, RMSE, MAPE) se elige el modelo de menor MAPE por serie y se pronostica el
siguiente mes con los 36 meses completos.
"""

import warnings

import numpy as np
import pandas as pd

MODELOS = ["Promedio móvil", "Holt", "Holt-Winters", "MLP"]
VENTANA_PM = 3
REZAGOS_MLP = 6
MESES_PRUEBA = 6


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------
def calcular_metricas(real, pronostico):
    real = np.asarray(real, dtype=float)
    pronostico = np.asarray(pronostico, dtype=float)
    error = real - pronostico
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    mascara = real != 0
    mape = float(np.mean(np.abs(error[mascara] / real[mascara])) * 100) if mascara.any() else np.nan
    return mae, rmse, mape


# ---------------------------------------------------------------------------
# Métodos (cada uno recibe la historia y devuelve el pronóstico del mes siguiente)
# ---------------------------------------------------------------------------
def promedio_movil(historia, fechas=None, ventana=VENTANA_PM):
    return float(np.mean(historia[-ventana:]))


def holt(historia, fechas=None):
    from statsmodels.tsa.holtwinters import Holt

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        modelo = Holt(np.asarray(historia, dtype=float), initialization_method="estimated").fit()
    return max(float(modelo.forecast(1)[0]), 0.0)


def holt_winters(historia, fechas):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    serie = pd.Series(np.asarray(historia, dtype=float), index=pd.DatetimeIndex(fechas, freq="MS"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        modelo = ExponentialSmoothing(
            serie, trend="add", seasonal="mul", seasonal_periods=12,
            initialization_method="estimated",
        ).fit(optimized=True)
    return max(float(modelo.forecast(1).iloc[0]), 0.0)


def _variables_mlp(valores, meses, i, rezagos):
    mes = meses[i]
    return list(valores[i - rezagos:i]) + [
        np.sin(2 * np.pi * mes / 12), np.cos(2 * np.pi * mes / 12), float(i),
    ]


def mlp(historia, fechas, rezagos=REZAGOS_MLP, semilla=42):
    """Perceptrón multicapa entrenado con ventanas deslizantes (sin mezclar el orden temporal)."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    valores = np.asarray(historia, dtype=float)
    fechas = pd.DatetimeIndex(fechas)
    siguiente = fechas[-1] + pd.DateOffset(months=1)
    meses = list(fechas.month) + [siguiente.month]

    X = np.array([_variables_mlp(valores, meses, i, rezagos) for i in range(rezagos, len(valores))])
    y = valores[rezagos:]
    x_nuevo = np.array([_variables_mlp(np.append(valores, np.nan), meses, len(valores), rezagos)])

    esc_x, esc_y = StandardScaler(), StandardScaler()
    Xs = esc_x.fit_transform(X)
    ys = esc_y.fit_transform(y.reshape(-1, 1)).ravel()
    red = MLPRegressor(
        hidden_layer_sizes=(32, 16), activation="relu", solver="adam",
        learning_rate_init=0.005, max_iter=200, shuffle=False, random_state=semilla,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        red.fit(Xs, ys)
    pred = esc_y.inverse_transform(red.predict(esc_x.transform(x_nuevo)).reshape(-1, 1))[0, 0]
    return max(float(pred), 0.0)


FUNCIONES = {
    "Promedio móvil": promedio_movil,
    "Holt": holt,
    "Holt-Winters": holt_winters,
    "MLP": mlp,
}


# ---------------------------------------------------------------------------
# Evaluación de todas las series
# ---------------------------------------------------------------------------
def evaluar_serie(valores, fechas, meses_prueba=MESES_PRUEBA, modelos=MODELOS):
    n = len(valores)
    inicio = n - meses_prueba
    pruebas, metricas, finales = [], [], {}
    siguiente = pd.Timestamp(fechas[-1]) + pd.DateOffset(months=1)
    for nombre in modelos:
        f = FUNCIONES[nombre]
        preds = []
        for t in range(inicio, n):
            p = f(valores[:t], fechas[:t])
            preds.append(p)
            pruebas.append({"Fecha": fechas[t], "Modelo": nombre, "Real": valores[t], "Pronóstico": p})
        mae, rmse, mape = calcular_metricas(valores[inicio:], preds)
        metricas.append({"Modelo": nombre, "MAE": mae, "RMSE": rmse, "MAPE": mape})
        finales[nombre] = f(valores, fechas)
    return pruebas, metricas, finales, siguiente


def evaluar_todo(consolidada, meses_prueba=MESES_PRUEBA, progreso=None):
    """Devuelve (pruebas, métricas, pronósticos por modelo, pronóstico final seleccionado)."""
    filas_prueba, filas_metricas, filas_pron = [], [], []
    grupos = list(consolidada.groupby(["Producto", "Cliente"]))
    for k, ((prod, cli), g) in enumerate(grupos):
        g = g.sort_values("Fecha")
        valores = g["Demanda"].to_numpy(dtype=float)
        fechas = pd.DatetimeIndex(g["Fecha"])
        pruebas, metricas, finales, siguiente = evaluar_serie(valores, fechas, meses_prueba)
        for r in pruebas:
            filas_prueba.append({"Producto": prod, "Cliente": cli, **r})
        for r in metricas:
            filas_metricas.append({"Producto": prod, "Cliente": cli, **r})
        for m, v in finales.items():
            filas_pron.append({"Producto": prod, "Cliente": cli, "Fecha": siguiente, "Modelo": m, "Pronóstico": v})
        if progreso is not None:
            progreso((k + 1) / len(grupos), f"{prod} · {cli}")

    pruebas = pd.DataFrame(filas_prueba)
    metricas = pd.DataFrame(filas_metricas)
    pron_modelos = pd.DataFrame(filas_pron)

    # Selección automática: menor MAPE (en empate, menor RMSE)
    orden = metricas.sort_values(["Producto", "Cliente", "MAPE", "RMSE"])
    ganador = orden.groupby(["Producto", "Cliente"], as_index=False).first()
    ganador = ganador.rename(columns={"Modelo": "Modelo seleccionado"})
    final = ganador.merge(
        pron_modelos, left_on=["Producto", "Cliente", "Modelo seleccionado"],
        right_on=["Producto", "Cliente", "Modelo"], how="left",
    ).drop(columns=["Modelo"])
    final = final[["Producto", "Cliente", "Fecha", "Modelo seleccionado", "Pronóstico", "MAE", "RMSE", "MAPE"]]
    return pruebas, metricas, pron_modelos, final


def resumen_global(metricas):
    return (
        metricas.groupby("Modelo")[["MAE", "RMSE", "MAPE"]].mean()
        .reindex(MODELOS).reset_index().sort_values("MAPE").reset_index(drop=True)
    )


def tabla_por_producto(metricas, metrica="MAPE"):
    """Métrica promedio por producto y modelo, con el ganador de cada fila."""
    t = metricas.pivot_table(index="Producto", columns="Modelo", values=metrica, aggfunc="mean")
    t = t.reindex(columns=MODELOS)
    t["Ganador"] = t[MODELOS].idxmin(axis=1)
    return t.reset_index()
