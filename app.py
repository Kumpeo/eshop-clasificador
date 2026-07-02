"""
E-Shop Clothing 2008 — Clasificador de Intención de Compra
Taller de Aplicaciones · Magister Data Science · Universidad San Sebastián
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import io
warnings.filterwarnings("ignore")

from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from modelo import ClasificadorIntención
from visualizaciones import (
    fig_roc, fig_confusion, fig_modelos, fig_importancias,
    fig_distribucion_etiqueta, fig_clusters_pca, fig_silhouette
)

# ══════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN DE PÁGINA
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="E-Shop · Clasificador Intención de Compra",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.hero {
    background: linear-gradient(135deg, #1C3A5E 0%, #2E86AB 100%);
    border-radius: 12px; padding: 2rem 2.5rem;
    margin-bottom: 1.5rem; color: white;
}
.hero h1 { font-size: 1.9rem; margin: 0 0 .4rem; font-weight: 700; }
.hero p  { font-size: 1rem; opacity: .85; margin: 0; }
.result-box {
    border-radius: 12px; padding: 1.5rem;
    text-align: center; margin-top: .5rem; border: 2px solid;
}
.result-alta { background: #d1fae5; border-color: #10b981; }
.result-baja { background: #f8fafc; border-color: #cbd5e1; }
.result-title { font-size: 1.5rem; font-weight: 700; margin-bottom: .3rem; }
.result-prob  { font-size: 1rem; margin-bottom: .8rem; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  FUNCIÓN DATASET DEMO — definida ANTES de usarse
# ══════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def _generar_dataset_demo():
    rng = np.random.default_rng(42)
    n_sesiones = 5_000
    clicks_por_ses = rng.integers(1, 25, n_sesiones)
    records = []
    for sid, n_cl in enumerate(clicks_por_ses):
        cat   = rng.integers(1, 5, n_cl)
        color = rng.integers(1, 15, n_cl)
        price = rng.uniform(18, 82, n_cl)
        page  = rng.integers(1, 6, n_cl)
        order = np.arange(1, n_cl + 1)
        photo = rng.integers(1, 3, n_cl)
        mes   = rng.integers(4, 9, n_cl)
        p2    = (price > price.mean()).astype(int) + 1
        mc    = [f"M{rng.integers(1,220)}" for _ in range(n_cl)]
        for i in range(n_cl):
            records.append({
                "session ID": sid, "order": order[i],
                "page 1 (main category)": cat[i],
                "page 2 (clothing model)": mc[i],
                "colour": color[i], "price": round(float(price[i]), 2),
                "page": page[i], "model photography": photo[i],
                "month": mes[i], "price 2": p2[i],
            })
    return pd.DataFrame(records)


@st.cache_data(show_spinner=False)
def entrenar_modelo(df_json, algo, n_estimadores, max_depth, test_size):
    df  = pd.read_json(io.StringIO(df_json))
    clf = ClasificadorIntención()
    clf.df = df
    clf._construir_sesiones()
    metricas = clf.entrenar(
        algoritmo=algo,
        n_estimators=n_estimadores,
        max_depth=max_depth,
        test_size=test_size / 100,
    )
    return clf, metricas


# ══════════════════════════════════════════════════════════════════
#  ESTADO DE SESIÓN
# ══════════════════════════════════════════════════════════════════
if "clf"      not in st.session_state: st.session_state.clf      = None
if "metricas" not in st.session_state: st.session_state.metricas = None
if "sesiones" not in st.session_state: st.session_state.sesiones = None
if "entrenado" not in st.session_state: st.session_state.entrenado = False


# ══════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🛍️ E-Shop Classifier")
    st.markdown("**Taller de Aplicaciones**  \nMagister Data Science · USS")
    st.divider()

    st.markdown("### 📂 Datos")
    fuente = st.radio(
        "Fuente del dataset",
        ["Usar dataset de ejemplo (integrado)", "Subir CSV propio"],
        label_visibility="collapsed",
    )

    df_raw = None
    if fuente == "Subir CSV propio":
        csv_file = st.file_uploader(
            "e-shop_clothing_2008.csv", type=["csv"],
            help="Separador: punto y coma (;)"
        )
        if csv_file:
            df_raw = pd.read_csv(csv_file, sep=";")
            st.success(f"✅ {len(df_raw):,} clicks cargados")
    else:
        df_raw = _generar_dataset_demo()
        st.info("📊 Dataset demo (5.000 sesiones sintéticas)")

    st.divider()
    st.markdown("### 🤖 Algoritmo")
    algoritmo = st.selectbox(
        "Modelo",
        ["Gradient Boosting", "Random Forest", "Árbol de Decisión"],
        label_visibility="collapsed",
    )

    with st.expander("⚙️ Hiperparámetros"):
        n_est  = st.slider("N° estimadores",    50, 300, 100, 50)
        max_d  = st.slider("Profundidad máxima", 2,  10,   4)
        test_s = st.slider("% datos de test",   10,  40,  30,  5)

    entrenar_btn = st.button("▶ Entrenar modelo", type="primary",
                              use_container_width=True)

    if st.session_state.entrenado:
        m = st.session_state.metricas
        st.success(
            f"✅ Modelo listo  \n"
            f"AUC = **{m['auc']:.3f}**  \n"
            f"CV  = **{m['cv_mean']:.3f} ± {m['cv_std']:.3f}**"
        )

    st.divider()
    st.markdown(
        "<small>Dataset: E-Shop Clothing 2008 · UCI ML Repository  \n"
        "Magister Data Science · USS</small>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════
#  ENTRENAMIENTO
# ══════════════════════════════════════════════════════════════════
if entrenar_btn and df_raw is not None:
    with st.spinner("Entrenando modelo..."):
        algo_key = {"Gradient Boosting": "GB",
                    "Random Forest": "RF",
                    "Árbol de Decisión": "DT"}[algoritmo]
        clf, metricas = entrenar_modelo(
            df_raw.to_json(), algo_key, n_est, max_d, test_s
        )
        st.session_state.clf       = clf
        st.session_state.metricas  = metricas
        st.session_state.sesiones  = clf.sesiones
        st.session_state.entrenado = True
    st.rerun()
elif entrenar_btn and df_raw is None:
    st.sidebar.error("Primero carga los datos.")


# ══════════════════════════════════════════════════════════════════
#  HERO
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <h1>🛍️ Clasificador de Intención de Compra</h1>
  <p>E-Shop Clothing 2008 · Análisis por sesión · Gradient Boosting + Feature Engineering</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  PESTAÑAS
# ══════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Métricas del modelo",
    "📈 Distribuciones",
    "🔬 Variables",
    "🔮 Predecir sesión",
    "📦 Predicción masiva",
])


# ── TAB 1: MÉTRICAS ───────────────────────────────────────────────
with tab1:
    if not st.session_state.entrenado:
        st.info("👈 Carga el dataset y presiona **▶ Entrenar modelo** en el panel lateral.")
    else:
        m   = st.session_state.metricas
        ses = st.session_state.sesiones

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("AUC-ROC (CV 5-fold)", f"{m['cv_mean']:.3f}", f"± {m['cv_std']:.3f}")
        c2.metric("Accuracy (test)",     f"{m['accuracy']*100:.1f}%")
        c3.metric("AUC-ROC (test)",      f"{m['auc']:.3f}")
        n_alta = int(ses["alta_intencion"].sum())
        c4.metric("Alta intención",
                  f"{n_alta:,} / {len(ses):,}",
                  f"{ses['alta_intencion'].mean()*100:.1f}%")

        st.divider()
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("##### Curva ROC")
            st.pyplot(fig_roc(m), use_container_width=True)
        with col_b:
            st.markdown("##### Matriz de confusión")
            st.pyplot(fig_confusion(m), use_container_width=True)

        st.divider()
        st.markdown("##### Comparativa de algoritmos (AUC-ROC · CV 5-fold)")
        st.pyplot(fig_modelos(m["resultados_comparativa"]), use_container_width=True)

        with st.expander("📋 Reporte de clasificación completo"):
            st.code(m["reporte"], language="text")


# ── TAB 2: DISTRIBUCIONES ─────────────────────────────────────────
with tab2:
    if not st.session_state.entrenado:
        st.info("👈 Entrena el modelo primero.")
    else:
        ses = st.session_state.sesiones
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("##### Distribución de etiquetas")
            st.pyplot(fig_distribucion_etiqueta(ses), use_container_width=True)

            st.markdown("##### Clicks por sesión")
            fig_cl, ax_cl = plt.subplots(figsize=(5, 3))
            ax_cl.hist(ses["n_clicks"].clip(upper=30), bins=25,
                       color="#2E86AB", edgecolor="white", linewidth=.5)
            ax_cl.axvline(ses["n_clicks"].median(), color="#E63946",
                          linestyle="--", linewidth=1.5,
                          label=f"Mediana={ses['n_clicks'].median():.0f}")
            ax_cl.set_xlabel("N° clicks"); ax_cl.set_ylabel("Sesiones")
            ax_cl.legend(fontsize=9)
            ax_cl.spines[["top","right"]].set_visible(False)
            fig_cl.tight_layout()
            st.pyplot(fig_cl, use_container_width=True)

        with col_b:
            st.markdown("##### Precio medio por segmento")
            fig_pr, ax_pr = plt.subplots(figsize=(5, 3))
            ax_pr.hist(ses[ses["alta_intencion"]==1]["precio_medio"].clip(upper=80),
                       bins=20, alpha=.7, color="#2E86AB",
                       label="Alta intención", edgecolor="white")
            ax_pr.hist(ses[ses["alta_intencion"]==0]["precio_medio"].clip(upper=80),
                       bins=20, alpha=.7, color="#F18F01",
                       label="Baja intención", edgecolor="white")
            ax_pr.set_xlabel("Precio medio (USD)"); ax_pr.set_ylabel("Sesiones")
            ax_pr.legend(fontsize=9)
            ax_pr.spines[["top","right"]].set_visible(False)
            fig_pr.tight_layout()
            st.pyplot(fig_pr, use_container_width=True)

            st.markdown("##### % clicks en rebajas por segmento")
            fig_reb, ax_reb = plt.subplots(figsize=(5, 3))
            ax_reb.hist(ses[ses["alta_intencion"]==1]["pct_rebajas"], bins=15,
                        alpha=.7, color="#2E86AB", label="Alta", edgecolor="white")
            ax_reb.hist(ses[ses["alta_intencion"]==0]["pct_rebajas"], bins=15,
                        alpha=.7, color="#F18F01", label="Baja", edgecolor="white")
            ax_reb.set_xlabel("Proporción clicks en rebajas")
            ax_reb.set_ylabel("Sesiones"); ax_reb.legend(fontsize=9)
            ax_reb.spines[["top","right"]].set_visible(False)
            fig_reb.tight_layout()
            st.pyplot(fig_reb, use_container_width=True)

        with st.expander("📊 Estadísticas descriptivas por segmento"):
            cols_show = ["n_clicks","n_modelos_uniq","precio_medio",
                         "pct_rebajas","tasa_exploracion","engagement_score"]
            df_desc = ses.groupby("alta_intencion")[cols_show].mean().round(3)
            df_desc.index = ["Baja intención","Alta intención"]
            st.dataframe(df_desc, use_container_width=True)


# ── TAB 3: VARIABLES ──────────────────────────────────────────────
with tab3:
    if not st.session_state.entrenado:
        st.info("👈 Entrena el modelo primero.")
    else:
        m   = st.session_state.metricas
        ses = st.session_state.sesiones

        col_a, col_b = st.columns([3, 2])
        with col_a:
            st.markdown("##### Importancia de variables (Top 15)")
            st.pyplot(fig_importancias(m), use_container_width=True)
        with col_b:
            st.markdown("##### 5 nuevos atributos (Feature Engineering)")
            fe_info = [
                ("★ Tasa exploración",   "n_modelos / n_clicks",
                 "Productos distintos por click. Alto → usuario selectivo."),
                ("★ Concentración cat.", "max(pct_cat_i)",
                 "Dominancia de una categoría en la sesión."),
                ("★ Premium × Depth",   "pct_caro × max_page",
                 "Sesiones premium que llegan profundo."),
                ("★ Colores/Click",      "n_colores / n_clicks",
                 "Diversidad visual por unidad de navegación."),
                ("★ Engagement Score",  "(clicks/max + page/5)/2",
                 "Índice combinado de intensidad de sesión."),
            ]
            for nombre, formula, desc in fe_info:
                with st.container(border=True):
                    st.markdown(f"**{nombre}** — `{formula}`")
                    st.caption(desc)

        st.divider()
        st.markdown("##### Clustering de sesiones (PCA 2D)")
        col_p1, col_p2 = st.columns([2, 1])
        with col_p1:
            st.pyplot(fig_clusters_pca(ses), use_container_width=True)
        with col_p2:
            st.markdown("**Silhouette por K**")
            st.pyplot(fig_silhouette(ses), use_container_width=True)

        with st.expander("🔗 Correlación features vs etiqueta"):
            feats = ["n_categorias","n_modelos_uniq","n_colores_uniq",
                     "pct_rebajas","precio_medio","pct_sobre_prom",
                     "tasa_exploracion","concentracion_cat","colores_por_click"]
            corrs = ses[feats].corrwith(ses["alta_intencion"]).sort_values()
            fig_c, ax_c = plt.subplots(figsize=(6, 3.5))
            colors_c = ["#E63946" if v < 0 else "#2E86AB" for v in corrs]
            ax_c.barh(corrs.index, corrs.values, color=colors_c, edgecolor="white")
            ax_c.axvline(0, color="gray", linewidth=.8)
            ax_c.set_xlabel("Correlación de Pearson")
            ax_c.spines[["top","right"]].set_visible(False)
            fig_c.tight_layout()
            st.pyplot(fig_c, use_container_width=True)


# ── TAB 4: PREDICCIÓN INDIVIDUAL ──────────────────────────────────
with tab4:
    if not st.session_state.entrenado:
        st.info("👈 Entrena el modelo primero.")
    else:
        st.markdown("Ingresa los datos de una sesión para predecir su intención de compra.")
        col_form, col_res = st.columns([3, 2])

        with col_form:
            with st.form("form_prediccion"):
                st.markdown("##### Datos de la sesión")

                r1c1, r1c2, r1c3 = st.columns(3)
                with r1c1: n_cat = st.number_input("N° categorías",      1, 4,   2)
                with r1c2: n_mod = st.number_input("N° productos únicos", 1, 168, 5)
                with r1c3: n_col = st.number_input("N° colores distintos",1, 14,  3)

                r2c1, r2c2, r2c3 = st.columns(3)
                with r2c1: pct_reb = st.slider("% rebajas",    0.0, 1.0, 0.05, 0.05)
                with r2c2: pct_blu = st.slider("% blusas",     0.0, 1.0, 0.30, 0.05)
                with r2c3: pct_pan = st.slider("% pantalones", 0.0, 1.0, 0.40, 0.05)

                r3c1, r3c2, r3c3 = st.columns(3)
                with r3c1: pct_fal  = st.slider("% faldas",       0.0, 1.0, 0.25, 0.05)
                with r3c2: precio   = st.number_input("Precio medio (USD)", 10, 100, 44)
                with r3c3: rango_p  = st.number_input("Rango precios (USD)", 0, 80, 20)

                r4c1, r4c2, r4c3 = st.columns(3)
                with r4c1: pct_caro = st.slider("% precio alto",  0.0, 1.0, 0.50, 0.05)
                with r4c2: pct_foto = st.slider("% foto frente",  0.0, 1.0, 0.75, 0.05)
                with r4c3:
                    mes = st.selectbox("Mes", [4,5,6,7,8], index=2,
                                       format_func=lambda x: {4:"Abril",5:"Mayo",
                                                               6:"Junio",7:"Julio",
                                                               8:"Agosto"}[x])
                submitted = st.form_submit_button(
                    "🔮 Clasificar sesión", type="primary", use_container_width=True
                )

        with col_res:
            if submitted:
                clf = st.session_state.clf
                n_cl_est = 7
                datos = {
                    "n_categorias": n_cat, "n_modelos_uniq": n_mod,
                    "n_colores_uniq": n_col, "pct_rebajas": pct_reb,
                    "pct_blusas": pct_blu, "pct_pantalones": pct_pan,
                    "pct_faldas": pct_fal, "precio_medio": precio,
                    "precio_rango": rango_p, "pct_sobre_prom": pct_caro,
                    "pct_foto_frente": pct_foto, "mes": mes,
                    "tasa_exploracion":  n_mod / max(n_cl_est, 1),
                    "concentracion_cat": max(pct_reb, pct_blu, pct_pan, pct_fal),
                    "colores_por_click": n_col / max(n_cl_est, 1),
                }
                pred, prob = clf.predecir_sesion(datos)

                st.markdown("##### Resultado")
                if pred == 1:
                    st.markdown(f"""
                    <div class="result-box result-alta">
                      <div class="result-title" style="color:#065f46">🛒 ALTA INTENCIÓN</div>
                      <div class="result-prob">Probabilidad: <strong>{prob*100:.1f}%</strong></div>
                    </div>""", unsafe_allow_html=True)
                    st.progress(float(prob))
                    st.success("**Acciones recomendadas:**  \n"
                               "• Mostrar botón 'Comprar ahora' prominente  \n"
                               "• Activar descuento de urgencia  \n"
                               "• Ofrecer envío express gratuito  \n"
                               "• Pop-up de checkout simplificado")
                else:
                    st.markdown(f"""
                    <div class="result-box result-baja">
                      <div class="result-title" style="color:#475569">👀 BAJA INTENCIÓN</div>
                      <div class="result-prob">Probabilidad alta: <strong>{prob*100:.1f}%</strong></div>
                    </div>""", unsafe_allow_html=True)
                    st.progress(float(prob))
                    st.info("**Acciones recomendadas:**  \n"
                            "• Mostrar contenido inspiracional  \n"
                            "• Sugerir productos relacionados  \n"
                            "• Email de seguimiento personalizado  \n"
                            "• Pop-up de 'Te puede gustar'")

                with st.expander("🔍 Atributos derivados calculados"):
                    st.dataframe(pd.DataFrame([{
                        "tasa_exploracion":  f"{datos['tasa_exploracion']:.3f}",
                        "concentracion_cat": f"{datos['concentracion_cat']:.3f}",
                        "colores_por_click": f"{datos['colores_por_click']:.3f}",
                    }]), use_container_width=True)
            else:
                st.markdown("##### Resultado")
                st.markdown("""
                <div style="border:2px dashed #cbd5e1;border-radius:12px;
                            padding:2rem;text-align:center;color:#94a3b8">
                  <div style="font-size:2rem">?</div>
                  <div>Completa el formulario y presiona<br>
                  <strong>Clasificar sesión</strong></div>
                </div>""", unsafe_allow_html=True)


# ── TAB 5: PREDICCIÓN MASIVA ──────────────────────────────────────
with tab5:
    if not st.session_state.entrenado:
        st.info("👈 Entrena el modelo primero.")
    else:
        st.markdown("Sube un CSV con el mismo formato para predecir todas las sesiones.")

        lote_file = st.file_uploader(
            "Selecciona el CSV de lote", type=["csv"], key="lote_uploader",
            help="Mismo formato que e-shop_clothing_2008.csv (sep=;)"
        )

        if lote_file:
            with st.spinner("Procesando sesiones..."):
                clf = st.session_state.clf
                df_lote  = pd.read_csv(lote_file, sep=";")
                resultado = clf.predecir_lote(df_lote)

            n      = len(resultado)
            n_alta = (resultado["prediccion"] == "Alta Intención").sum()

            c1, c2, c3 = st.columns(3)
            c1.metric("Sesiones procesadas", f"{n:,}")
            c2.metric("Alta intención",  f"{n_alta:,}", f"{n_alta/n*100:.1f}%")
            c3.metric("Baja intención",  f"{n-n_alta:,}", f"{(n-n_alta)/n*100:.1f}%")

            st.divider()
            filtro = st.selectbox(
                "Mostrar",
                ["Todas las sesiones","Solo alta intención","Solo baja intención"]
            )
            df_show = resultado.copy()
            if filtro == "Solo alta intención":
                df_show = df_show[df_show["prediccion"]=="Alta Intención"]
            elif filtro == "Solo baja intención":
                df_show = df_show[df_show["prediccion"]=="Baja Intención"]

            def color_pred(val):
                if val == "Alta Intención":
                    return "background-color:#d1fae5;color:#065f46;font-weight:600"
                return "color:#475569"

            st.dataframe(
                df_show.style.map(color_pred, subset=["prediccion"])
                       .format({"probabilidad_alta_intencion": "{:.1%}"}),
                use_container_width=True, height=380
            )

            csv_out = resultado.to_csv(index=False, sep=";").encode("utf-8-sig")
            st.download_button(
                "💾 Descargar resultados (CSV)", data=csv_out,
                file_name="predicciones_lote.csv", mime="text/csv", type="primary"
            )
