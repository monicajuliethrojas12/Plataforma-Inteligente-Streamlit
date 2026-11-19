# Plataforma inteligente para la planeación de la producción y la distribución

Tablero en Streamlit que integra la analítica predictiva (ETL, análisis descriptivo,
cuatro modelos de pronóstico y selección automática) con la analítica prescriptiva
(modelo de programación lineal en PuLP y análisis de escenarios).

## Cómo ejecutarlo

1. Instalar las librerías (una sola vez):

   ```
   pip install -r requirements.txt
   ```

2. Abrir una terminal en esta carpeta y ejecutar:

   ```
   streamlit run app.py
   ```

3. En el panel izquierdo cargar **Datos.xlsx** (o usar el botón *Usar el Datos.xlsx incluido*).
   La primera carga entrena los cuatro modelos en las 50 series y toma cerca de medio minuto;
   después queda guardada y la aplicación responde de inmediato.

## Datos

- `Datos.xlsx`: productos, clientes, plantas, nodos, capacidades y costos de transporte.
  Hojas obligatorias: Producto, Cliente, Planta, Nodo, capacidad_planta_producto,
  costo_transporte_planta_nodo y costo_distribucion_nodo_cliente.
- Demanda histórica: si el libro no trae una hoja con Fecha, Producto, Cliente y Demanda,
  la aplicación usa la base de la Fase 1 (36 meses, 1.800 registros, semilla 42).
  También se puede cargar un CSV o Excel aparte en *Demanda histórica (opcional)*;
  en `insumos/demanda_historica_plana.csv` está la base completa.

## Archivos

| Archivo | Contenido |
|---|---|
| `app.py` | Tablero: secciones, gráficos e integración de todos los módulos |
| `escenarios.py` | Escenarios base, optimista y pesimista, y tablas comparativas |
| `datos.py` | Lectura de Datos.xlsx y de la demanda histórica |
| `etl.py` | Limpieza, verificación, consolidación y estadísticas descriptivas |
| `pronosticos.py` | Promedio móvil, Holt, Holt-Winters, MLP, métricas y selección del mejor modelo |
| `optimizacion.py` | Modelo de costo mínimo de producción, trasbordo y distribución (PuLP) |
| `graficos.py` | Estilo visual, diagrama de la red y gráficos reutilizables |
| `insumos/` | Datos.xlsx y la demanda histórica en CSV |

## Resultados de referencia (escenario base, noviembre de 2026)

| Modelo | MAE | RMSE | MAPE |
|---|---|---|---|
| Holt-Winters | 20,13 | 24,32 | 6,73 % |
| MLP | 33,53 | 39,57 | 11,44 % |
| Holt | 53,24 | 63,68 | 18,55 % |
| Promedio móvil | 65,97 | 78,33 | 25,13 % |

Demanda pronosticada: 16.264 unidades. Costo total mínimo: $ 1.338.493.450 COP.
