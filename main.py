import tensorflow as tf
import numpy as np
import mysql.connector, os
from fastapi import FastAPI, Query
from pydantic import BaseModel
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()


def get_db_connection():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME")
    )

model = None

def load_model():
    global model
    try:
        print("cargando modelo...")
        model = tf.keras.models.load_model("modelo_estres_dinamico.h5")
        print("modelo cargado correctamente.")
    except Exception as e:
        print("error al cargar el modelo:", e)

# Obtener datos históricos de la empresa
def obtener_datos_historicos(empresa_id: int):
    try:
        print(f"obteniendo datos para empresa_id: {empresa_id}")  # Debug

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT DATE(s.created_at) as fecha, ROUND(AVG(s.caritas), 2) as promedio_caritas
            FROM user_estres_sessions s
            JOIN users u ON s.user_id = u.id
            WHERE s.caritas IS NOT NULL AND u.empresa_id = %s
            GROUP BY fecha
            ORDER BY fecha ASC;
        """, (empresa_id,))
        datos = cursor.fetchall()
        conn.close()

        if not datos:
            print(f"no hay datos históricos para la empresa {empresa_id}.")
            return {"error": f"no hay datos de estrés registrados para la empresa {empresa_id}."}

        return datos
    except Exception as e:
        print("error al obtener datos históricos:", e)
        return {"error": "error interno al obtener datos históricos."}



def calcular_mae(valores_reales, valores_predichos):
    errores = [abs(real - predicho) for real, predicho in zip(valores_reales, valores_predichos)]
    return sum(errores) / len(errores) if errores else 0

app = FastAPI()

@app.get("/predict")
def predecir_estres(id_empresa: int = Query(..., description="ID de la empresa")):
    global model
    if model is None:
        load_model()
        if model is None:
            return {"error": "No se pudo cargar el modelo"}

    historial = obtener_datos_historicos(id_empresa)

    if isinstance(historial, dict) and "error" in historial:
        return historial  

    caritas_values = [float(row["promedio_caritas"]) for row in historial]

    if len(caritas_values) < 7:
        return {
            "historico": historial,
            "prediccion": [],
            "mensaje": "Se necesitan al menos 7 días de datos."
        }

    input_sequence = caritas_values[-7:]
    caritas_normalizadas = [(valor - 1) / (3 - 1) for valor in input_sequence]
    X_input = np.array(caritas_normalizadas).reshape(1, 7, 1)

    prediccion_tensor = model.predict(X_input)
    valores_predichos_normalizados = prediccion_tensor.flatten()
    valores_predichos_escalados = [(valor * (3 - 1)) + 1 for valor in valores_predichos_normalizados]

    ultima_fecha = datetime.strptime(str(historial[-1]["fecha"]), "%Y-%m-%d")
    predicciones = [{"fecha": (ultima_fecha + timedelta(days=i+1)).strftime("%Y-%m-%d"), "caritas_predicho": round(valor, 2)}
                    for i, valor in enumerate(valores_predichos_escalados)]

    historico_real = caritas_values[-7:]
    prediccion_ultimos_7_dias = valores_predichos_escalados[:7]
    mae = calcular_mae(historico_real, prediccion_ultimos_7_dias)
    precision = max(0, 100 - ((mae / (3 - 1)) * 100))

    return {
        "historico": [{"fecha": row["fecha"], "caritas": row["promedio_caritas"]} for row in historial],
        "prediccion": predicciones,
        "precision": f"{precision:.2f}%",
        "id_empresa": id_empresa
    }

if __name__ == "__main__":
    import uvicorn
    load_model()
    uvicorn.run(app, host="0.0.0.0", port=8000) 
