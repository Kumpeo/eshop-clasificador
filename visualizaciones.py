"""
visualizaciones.py
Funciones que devuelven figuras matplotlib listas para st.pyplot().
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, silhouette_score, davies_bouldin_score

# ── Paleta ──────────────────────────────────────────────────────
C1 = "#1C3A5E"
C2 = "#2E86AB"
C3 = "#A23B72"
C4 = "#F18F01"
C5 = "#2DC653"
GRAY = "#64748B"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linewidth":    0.6,
})


# ── 1. Curva ROC ─────────────────────────────────────────────────
def fig_roc(m):
    fpr, tpr, _ = roc_curve(m["y_te"], m["y_prob"])
    fig, ax = plt.subplots(figsize=(5, 3.8))
    ax.plot(fpr, tpr, color=C2, linewidth=2.5, label=f"AUC = {m['auc']:.3f}")
    ax.fill_between(fpr, tpr, alpha=0.08, color=C2)
    ax.plot([0, 1], [0, 1], "--", color=GRAY, linewidth=1, label="Azar (0.5)")
    ax.set_xlabel("Tasa falsos positivos")
    ax.set_ylabel("Tasa verdaderos positivos")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


# ── 2. Matriz de confusión ────────────────────────────────────────
def fig_confusion(m):
    cm = m["cm"]
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Baja", "Alta"],
                yticklabels=["Baja", "Alta"],
                annot_kws={"size": 14, "weight": "bold"},
                linewidths=2, linecolor="white")
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    sub_labels = ["TN", "FP", "FN", "TP"]
    for txt, sub in zip(ax.texts, sub_labels):
        txt.set_text(f"{txt.get_text()}\n({sub})")
    fig.tight_layout()
    return fig


# ── 3. Comparativa de modelos ─────────────────────────────────────
def fig_modelos(comparativa):
    nombres = list(comparativa.keys())
    medias  = [comparativa[n]["cv_mean"] for n in nombres]
    stds    = [comparativa[n]["cv_std"]  for n in nombres]
    colores = [C2, C5, C3]

    fig, ax = plt.subplots(figsize=(6, 2.8))
    bars = ax.barh(nombres, medias, xerr=stds, color=colores[:len(nombres)],
                   edgecolor="white", capsize=5, height=0.45)
    ax.set_xlim(0.80, 1.01)
    ax.set_xlabel("AUC-ROC (CV 5-fold)")
    for bar, v in zip(bars, medias):
        ax.text(v + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=10, fontweight="bold")
    fig.tight_layout()
    return fig


# ── 4. Importancia de variables ───────────────────────────────────
def fig_importancias(m):
    imp = m["importancias"].head(15).sort_values(ascending=True)
    etiq = {
        "n_categorias":    "N° categorías",
        "n_modelos_uniq":  "Prod. únicos",
        "n_colores_uniq":  "N° colores",
        "pct_rebajas":     "% Rebajas",
        "pct_blusas":      "% Blusas",
        "pct_pantalones":  "% Pantalones",
        "pct_faldas":      "% Faldas",
        "precio_medio":    "Precio medio",
        "precio_rango":    "Rango precios",
        "pct_sobre_prom":  "% Precio alto",
        "pct_foto_frente": "% Foto frente",
        "mes":             "Mes",
        "tasa_exploracion":  "★ Tasa exploración",
        "concentracion_cat": "★ Conc. categoría",
        "colores_por_click": "★ Colores/Click",
    }
    fe_nuevas = {"tasa_exploracion", "concentracion_cat", "colores_por_click"}
    lbl_imp = [etiq.get(v, v) for v in imp.index]
    col_imp = [C4 if v in fe_nuevas else C2 for v in imp.index]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(lbl_imp, imp.values, color=col_imp, edgecolor="white")
    ax.set_xlabel("Importancia relativa")
    for i, v in enumerate(imp.values):
        ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=8.5)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=C4, label="★ Nuevo atributo (FE)"),
        Patch(facecolor=C2, label="Atributo original"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="lower right")
    fig.tight_layout()
    return fig


# ── 5. Distribución de etiqueta ───────────────────────────────────
def fig_distribucion_etiqueta(sesiones):
    vc = sesiones["alta_intencion"].value_counts()
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    bars = ax.bar(["Baja intención", "Alta intención"],
                  vc.sort_index().values,
                  color=[GRAY, C5], edgecolor="white", width=0.5)
    ax.set_ylabel("N° sesiones")
    for bar, v in zip(bars, vc.sort_index().values):
        pct = v / len(sesiones) * 100
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 50,
                f"{v:,}\n({pct:.1f}%)",
                ha="center", fontsize=9, fontweight="bold")
    fig.tight_layout()
    return fig


# ── 6. Clusters PCA 2D ────────────────────────────────────────────
def fig_clusters_pca(sesiones):
    features_cl = ["n_clicks","n_categorias","n_modelos_uniq","pct_rebajas",
                   "precio_medio","precio_rango","pct_sobre_prom","max_page_sitio"]
    X = StandardScaler().fit_transform(sesiones[features_cl].fillna(0))

    K = 5
    labels = KMeans(n_clusters=K, random_state=42, n_init=10).fit_predict(X)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    var = pca.explained_variance_ratio_.sum() * 100

    fig, ax = plt.subplots(figsize=(5.5, 4))
    colores = plt.cm.tab10(np.linspace(0, 1, K))
    for c in range(K):
        mask = labels == c
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=[colores[c]], alpha=0.3, s=6, label=f"Cl.{c}")
    ax.set_title(f"K-Means K=5 · PCA (varianza={var:.1f}%)", fontsize=10)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(markerscale=4, fontsize=8)
    fig.tight_layout()
    return fig


# ── 7. Silhouette por método ──────────────────────────────────────
def fig_silhouette(sesiones):
    features_cl = ["n_clicks","n_categorias","n_modelos_uniq","pct_rebajas",
                   "precio_medio","precio_rango","pct_sobre_prom","max_page_sitio"]
    X = StandardScaler().fit_transform(sesiones[features_cl].fillna(0))

    sils = []
    ks   = range(2, 8)
    for k in ks:
        lbl = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
        sils.append(silhouette_score(X, lbl, sample_size=min(5000, len(X))))

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(ks, sils, "o-", color=C2, linewidth=2, markersize=6)
    best_k = list(ks)[sils.index(max(sils))]
    ax.axvline(best_k, color=C3, linestyle="--", linewidth=1.5,
               label=f"Mejor K={best_k}")
    ax.set_xlabel("K")
    ax.set_ylabel("Silhouette")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig
