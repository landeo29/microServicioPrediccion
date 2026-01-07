import numpy as np
import joblib
import pymysql
import os
from fastapi import FastAPI, Query
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()


def get_db_connection():
    return pymysql.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        cursorclass=pymysql.cursors.DictCursor
    )


modelo_data = None


def load_model():
    global modelo_data
    try:
        print("Cargando modelo Ridge...")
        modelo_data = joblib.load("modelo_estres.joblib")
        print("Modelo cargado correctamente.")
    except Exception as e:
        print("Error al cargar el modelo:", e)



def obtener_datos_historicos(empresa_id: int):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT DATE(s.created_at) as fecha, ROUND(AVG(s.caritas), 2) as promedio_caritas,
                   COUNT(*) as total_sesiones, AVG(HOUR(s.created_at)) as hora_promedio
            FROM user_estres_sessions s
            JOIN users u ON s.user_id = u.id
            WHERE s.caritas IS NOT NULL 
              AND s.caritas BETWEEN 1 AND 3
              AND u.empresa_id = %s
            GROUP BY DATE(s.created_at)
            ORDER BY fecha ASC;
        """, (empresa_id,))
        datos = cursor.fetchall()
        conn.close()

        if not datos:
            return {"error": f"No hay datos de estrés para la empresa {empresa_id}."}
        return datos
    except Exception as e:
        print("Error al obtener datos:", e)
        return {"error": "Error interno al obtener datos."}



def preparar_features(historial):
    df_data = []
    for row in historial:
        df_data.append({
            'fecha': row['fecha'],
            'promedio_caritas': float(row['promedio_caritas']),
            'total_sesiones': row.get('total_sesiones', 1),
            'hora_promedio': row.get('hora_promedio', 12)
        })

    n = len(df_data)
    if n < 7:
        return None

    ultimo = df_data[-1]

    features = {
        'total_sesiones': ultimo['total_sesiones'],
        'hora_promedio': ultimo['hora_promedio'],
        'dia_semana': ultimo['fecha'].weekday() if hasattr(ultimo['fecha'], 'weekday') else datetime.strptime(
            str(ultimo['fecha']), '%Y-%m-%d').weekday(),
        'es_fin_semana': 1 if (ultimo['fecha'].weekday() if hasattr(ultimo['fecha'], 'weekday') else datetime.strptime(
            str(ultimo['fecha']), '%Y-%m-%d').weekday()) >= 5 else 0,
        'dia_mes': ultimo['fecha'].day if hasattr(ultimo['fecha'], 'day') else datetime.strptime(str(ultimo['fecha']),
                                                                                                 '%Y-%m-%d').day,
        'caritas_lag_1': df_data[-2]['promedio_caritas'] if n >= 2 else df_data[-1]['promedio_caritas'],
        'caritas_lag_2': df_data[-3]['promedio_caritas'] if n >= 3 else df_data[-1]['promedio_caritas'],
        'caritas_lag_3': df_data[-4]['promedio_caritas'] if n >= 4 else df_data[-1]['promedio_caritas'],
        'caritas_lag_5': df_data[-6]['promedio_caritas'] if n >= 6 else df_data[-1]['promedio_caritas'],
        'caritas_lag_7': df_data[-8]['promedio_caritas'] if n >= 8 else df_data[-1]['promedio_caritas'],
        'media_movil_3': np.mean([d['promedio_caritas'] for d in df_data[-3:]]),
        'media_movil_5': np.mean([d['promedio_caritas'] for d in df_data[-5:]]) if n >= 5 else np.mean(
            [d['promedio_caritas'] for d in df_data]),
        'media_movil_7': np.mean([d['promedio_caritas'] for d in df_data[-7:]]) if n >= 7 else np.mean(
            [d['promedio_caritas'] for d in df_data]),
        'tendencia': df_data[-1]['promedio_caritas'] - df_data[-2]['promedio_caritas'] if n >= 2 else 0
    }

    return features



def clasificar_estres(valor):
    valor_redondeado = int(round(valor))
    valor_redondeado = max(1, min(3, valor_redondeado))

    niveles = {
        1: {'nivel': 'Bajo', 'emoji': '😊'},
        2: {'nivel': 'Medio', 'emoji': '😐'},
        3: {'nivel': 'Alto', 'emoji': '😰'}
    }
    return {'valor': valor_redondeado, **niveles[valor_redondeado]}

def predecir(features, dias=3):
    global modelo_data

    model = modelo_data['model']
    scaler = modelo_data['scaler']
    feature_names = modelo_data['feature_names']

    X = np.array([[features.get(f, 0) for f in feature_names]])
    X_scaled = scaler.transform(X)

    predicciones = []
    features_temp = features.copy()

    for i in range(dias):
        pred = model.predict(scaler.transform(np.array([[features_temp.get(f, 0) for f in feature_names]])))[0]
        pred = np.clip(pred, 1, 3)
        predicciones.append(pred)

        features_temp['caritas_lag_7'] = features_temp['caritas_lag_5']
        features_temp['caritas_lag_5'] = features_temp['caritas_lag_3']
        features_temp['caritas_lag_3'] = features_temp['caritas_lag_2']
        features_temp['caritas_lag_2'] = features_temp['caritas_lag_1']
        features_temp['caritas_lag_1'] = pred
        features_temp['tendencia'] = pred - features_temp['caritas_lag_2']

    return predicciones


app = FastAPI()


@app.get("/predict")
def predecir_estres(id_empresa: int = Query(..., description="ID de la empresa")):
    global modelo_data

    if modelo_data is None:
        load_model()
        if modelo_data is None:
            return {"error": "No se pudo cargar el modelo"}

    historial = obtener_datos_historicos(id_empresa)
    if isinstance(historial, dict) and "error" in historial:
        return historial

    if len(historial) < 7:
        return {
            "puede_predecir": False,
            "historico": [{"fecha": str(row["fecha"]), "caritas": row["promedio_caritas"]} for row in historial],
            "prediccion": [],
            "dias_actuales": len(historial),
            "dias_necesarios": 7,
            "mensaje": f"Faltan {7 - len(historial)} días para activar predicción."
        }

    features = preparar_features(historial)

    ultima_fecha = datetime.strptime(str(historial[-1]["fecha"]), "%Y-%m-%d")
    valores_predichos = predecir(features, dias=3)

    predicciones = []
    for i, valor in enumerate(valores_predichos):
        fecha = (ultima_fecha + timedelta(days=i + 1)).strftime("%Y-%m-%d")
        info = clasificar_estres(valor)
        predicciones.append({
            "fecha": fecha,
            "valor": info['valor'],
            "nivel": info['nivel'],
            "emoji": info['emoji']
        })

    return {
        "puede_predecir": True,
        "historico": [{"fecha": str(row["fecha"]), "caritas": row["promedio_caritas"]} for row in historial],
        "prediccion": predicciones,
        "precision": f"{modelo_data['metrics']['r2'] * 100:.2f}%",
        "id_empresa": id_empresa
    }


if __name__ == "__main__":
    import uvicorn

    load_model()
    uvicorn.run(app, host="0.0.0.0", port=8000)