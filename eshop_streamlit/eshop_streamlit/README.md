# 🛍️ E-Shop Clothing 2008 — Clasificador de Intención de Compra

**Taller de Aplicaciones · Magister Data Science · Universidad San Sebastián**

Aplicación Streamlit que entrena y visualiza un clasificador de alta intención de compra
sobre el dataset E-Shop Clothing 2008 (UCI Machine Learning Repository).

---

## 🚀 Despliegue en Streamlit Community Cloud

### Paso 1 — Publicar en GitHub

```bash
# En la carpeta del proyecto:
git init
git add .
git commit -m "feat: clasificador intención de compra e-shop"
git branch -M main
git remote add origin https://github.com/TU_USUARIO/eshop-clasificador.git
git push -u origin main
```

### Paso 2 — Crear la app en Streamlit Cloud

1. Ve a **[share.streamlit.io](https://share.streamlit.io)** e inicia sesión con GitHub.
2. Click en **"New app"**.
3. Selecciona el repositorio y la rama `main`.
4. **Main file path:** `app.py`
5. Click **"Deploy!"**

La app estará disponible en:
```
https://TU_USUARIO-eshop-clasificador-app-XXXX.streamlit.app
```

---

## 💻 Ejecución local

```bash
# Crear y activar entorno
python -m venv venv_taller
.\venv_taller\Scripts\Activate.ps1          # Windows PowerShell
# source venv_taller/bin/activate           # macOS/Linux

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar
streamlit run app.py
```

---

## 📁 Estructura del proyecto

```
eshop_streamlit/
├── app.py                  ← Aplicación principal Streamlit
├── modelo.py               ← Pipeline: sesiones → features → clasificador
├── visualizaciones.py      ← Figuras matplotlib (ROC, confusión, importancias...)
├── requirements.txt        ← Dependencias para Streamlit Cloud
├── .streamlit/
│   └── config.toml         ← Tema oscuro y configuración del servidor
└── README.md               ← Este archivo
```

---

## 🗂️ Dataset

El dataset `e-shop_clothing_2008.csv` **no se incluye en el repositorio** (peso >10 MB).

La app incluye un **dataset demo sintético** (5.000 sesiones) para que funcione
sin subir el CSV. Si quieres usar el dataset original, cárgalo desde el sidebar
con la opción "Subir CSV propio".

Descarga el dataset original en:
[UCI ML Repository](https://archive.ics.uci.edu/dataset/553/online+shoppers+purchasing+intention+dataset)

---

## 📊 Funcionalidades

| Pestaña | Contenido |
|---|---|
| **Métricas del modelo** | AUC-ROC, Accuracy, Curva ROC, Matriz de confusión, comparativa de algoritmos |
| **Distribuciones** | Etiquetas, clicks por sesión, precio y rebajas por segmento |
| **Variables** | Importancia de features, clustering PCA, Silhouette, correlaciones |
| **Predecir sesión** | Formulario interactivo con resultado + recomendaciones de negocio |
| **Predicción masiva** | Carga un CSV y descarga predicciones para todas las sesiones |

---

## 🧠 Modelo

- **Algoritmo:** Gradient Boosting (también RF y Árbol de Decisión)
- **Features:** 15 variables (12 originales + 3 de Feature Engineering)
- **Variable objetivo:** Alta intención de compra (sesión larga + página profunda)
- **Sin data leakage:** `n_clicks` y `max_page_sitio` definen la etiqueta y no son features
- **AUC-ROC CV 5-fold:** ~0.966

---

## 📚 Referencias

- Łapczyński M., Białowąs S. (2013). Discovering Patterns of Users' Behaviour in an E-shop.
- Wani, A.A. (2025). Comprehensive review of dimensionality reduction algorithms. PeerJ CS.
- Li et al. (2025). ML from a Universe of signals: The role of feature engineering. J.Financial Economics.
