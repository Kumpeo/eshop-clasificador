"""
modelo.py
Pipeline de datos y clasificador reutilizable.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    confusion_matrix, classification_report
)


class ClasificadorIntención:
    """Encapsula el pipeline completo: sesiones → features → modelo → predicción."""

    FEATURES = [
        "n_categorias", "n_modelos_uniq", "n_colores_uniq",
        "pct_rebajas", "pct_blusas", "pct_pantalones", "pct_faldas",
        "precio_medio", "precio_rango", "pct_sobre_prom",
        "pct_foto_frente", "mes",
        "tasa_exploracion", "concentracion_cat", "colores_por_click",
    ]

    def __init__(self):
        self.df       = None
        self.sesiones = None
        self.modelo   = None
        self.metricas = {}
        self.entrenado = False

    # ── 1. Construcción de sesiones ───────────────────────────────
    def _construir_sesiones(self):
        df = self.df
        s = df.groupby("session ID").agg(
            n_clicks        = ("order", "count"),
            max_page_sitio  = ("page", "max"),
            n_paginas_sitio = ("page", "nunique"),
            n_categorias    = ("page 1 (main category)", "nunique"),
            n_modelos_uniq  = ("page 2 (clothing model)", "nunique"),
            n_colores_uniq  = ("colour", "nunique"),
            pct_rebajas     = ("page 1 (main category)", lambda x: (x == 4).mean()),
            pct_blusas      = ("page 1 (main category)", lambda x: (x == 3).mean()),
            pct_pantalones  = ("page 1 (main category)", lambda x: (x == 1).mean()),
            pct_faldas      = ("page 1 (main category)", lambda x: (x == 2).mean()),
            precio_medio    = ("price", "mean"),
            precio_rango    = ("price", lambda x: x.max() - x.min()),
            pct_sobre_prom  = ("price 2", lambda x: (x == 1).mean()),
            pct_foto_frente = ("model photography", lambda x: (x == 1).mean()),
            mes             = ("month", "first"),
        ).reset_index()

        # Feature Engineering — 5 nuevos atributos
        s["tasa_exploracion"]  = s["n_modelos_uniq"] / s["n_clicks"].clip(lower=1)
        s["concentracion_cat"] = s[["pct_rebajas","pct_blusas",
                                     "pct_pantalones","pct_faldas"]].max(axis=1)
        s["colores_por_click"] = s["n_colores_uniq"] / s["n_clicks"].clip(lower=1)
        # (premium_depth y engagement_score se usan para clustering, no para clf)
        s["premium_depth"]    = s["pct_sobre_prom"] * s["max_page_sitio"]
        s["engagement_score"] = (
            s["n_clicks"] / s["n_clicks"].max() +
            s["max_page_sitio"] / 5
        ) / 2

        # Etiqueta — sin data leakage:
        # n_clicks y max_page_sitio definen la etiqueta → no están en FEATURES
        p60 = s["n_clicks"].quantile(0.60)
        s["alta_intencion"] = (
            (s["n_clicks"] >= p60) & (s["max_page_sitio"] >= 3)
        ).astype(int)
        s["sensible_precio"] = (s["pct_rebajas"] >= 0.30).astype(int)

        self.sesiones = s

    # ── 2. Entrenamiento ──────────────────────────────────────────
    def entrenar(self, algoritmo="GB", n_estimators=100,
                 max_depth=4, test_size=0.30):

        self._construir_sesiones()
        s = self.sesiones

        X = s[self.FEATURES].fillna(0)
        y = s["alta_intencion"]

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Modelo principal
        if algoritmo == "GB":
            self.modelo = GradientBoostingClassifier(
                n_estimators=n_estimators, max_depth=max_depth, random_state=42
            )
        elif algoritmo == "RF":
            self.modelo = RandomForestClassifier(
                n_estimators=n_estimators, max_depth=max_depth,
                random_state=42, n_jobs=-1
            )
        else:  # DT
            self.modelo = DecisionTreeClassifier(
                max_depth=max_depth, random_state=42
            )

        self.modelo.fit(X_tr, y_tr)

        y_pred = self.modelo.predict(X_te)
        y_prob = self.modelo.predict_proba(X_te)[:, 1]

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.modelo, X, y, cv=cv,
                                    scoring="roc_auc", n_jobs=-1)

        cm = confusion_matrix(y_te, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # Comparativa de los 3 modelos (para el gráfico de tab1)
        comparativa = {}
        for nom, mod in [
            ("Árbol de\nDecisión", DecisionTreeClassifier(max_depth=max_depth, random_state=42)),
            ("Random\nForest",     RandomForestClassifier(n_estimators=n_estimators,
                                                           max_depth=max_depth,
                                                           random_state=42, n_jobs=-1)),
            ("Gradient\nBoosting", GradientBoostingClassifier(n_estimators=n_estimators,
                                                               max_depth=max_depth,
                                                               random_state=42)),
        ]:
            cs = cross_val_score(mod, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
            comparativa[nom] = {"cv_mean": cs.mean(), "cv_std": cs.std()}

        self.metricas = {
            "accuracy":    accuracy_score(y_te, y_pred),
            "auc":         roc_auc_score(y_te, y_prob),
            "cv_mean":     cv_scores.mean(),
            "cv_std":      cv_scores.std(),
            "cm":          cm,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "y_te":        y_te.values,
            "y_prob":      y_prob,
            "importancias": pd.Series(
                self.modelo.feature_importances_, index=self.FEATURES
            ).sort_values(ascending=False),
            "reporte": classification_report(
                y_te, y_pred,
                target_names=["Baja Intención", "Alta Intención"]
            ),
            "resultados_comparativa": comparativa,
            "algoritmo": algoritmo,
        }
        self.entrenado = True
        return self.metricas

    # ── 3. Predicción individual ──────────────────────────────────
    def predecir_sesion(self, datos_dict):
        if not self.entrenado:
            raise RuntimeError("El modelo no ha sido entrenado.")
        fila = pd.DataFrame([datos_dict])[self.FEATURES].fillna(0)
        prob = float(self.modelo.predict_proba(fila)[0][1])
        pred = int(prob >= 0.50)
        return pred, prob

    # ── 4. Predicción masiva ──────────────────────────────────────
    def predecir_lote(self, df_nuevo):
        tmp = ClasificadorIntención()
        tmp.df = df_nuevo
        tmp._construir_sesiones()
        X = tmp.sesiones[self.FEATURES].fillna(0)
        probs = self.modelo.predict_proba(X)[:, 1]
        preds = (probs >= 0.50).astype(int)
        res = tmp.sesiones[["session ID"]].copy()
        res["probabilidad_alta_intencion"] = probs
        res["prediccion"] = np.where(preds == 1, "Alta Intención", "Baja Intención")
        return res
