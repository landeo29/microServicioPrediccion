import tensorflow as tf
import numpy as np
import mysql.connector, os
from fastapi import FastAPI
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
        print("Cargando modelo...")
        model = tf.keras.models.load_model("modelo_estres_dinamico.h5")
        print("Modelo cargado correctamente.")
    except Exception as e:
        print("Error al cargar el modelo:", e)

def obtener_datos_historicos():
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT DATE(created_at) as fecha, ROUND(AVG(caritas), 2) as promedio_caritas
            FROM user_estres_sessions
            WHERE caritas IS NOT NULL
            GROUP BY fecha
            ORDER BY fecha ASC;
        """)
        datos = cursor.fetchall()
        conn.close()

        if len(datos) < 7:
            print("No hay suficientes datos históricos para hacer predicciones.")
            return None

        for row in datos:
            row["promedio_caritas"] = float(row["promedio_caritas"])

        return datos
    except Exception as e:
        print("Error al obtener datos históricos:", e)
        return None


def calcular_mae(valores_reales, valores_predichos):
    errores = [abs(real - predicho) for real, predicho in zip(valores_reales, valores_predichos)]
    return sum(errores) / len(errores)

app = FastAPI()

@app.get("/fastapi/predict")
def predecir_estres(): 
    global model
    if model is None:
        load_model()
        if model is None:
            return {"error": "No se pudo cargar el modelo"}

    historial = obtener_datos_historicos()
    if not historial:
        return {"error": "No hay suficientes datos históricos"}

    caritas_values = [row["promedio_caritas"] for row in historial]

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
        "precision": f"{precision:.2f}%"
    }

if __name__ == "__main__":
    import uvicorn
    load_model()
    uvicorn.run(app, host="0.0.0.0", port=8000)
