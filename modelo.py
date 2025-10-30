import pandas as pd
import numpy as np
import gdown
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors
import streamlit as st


# -----------------------------
# Descarga y carga del dataset
# -----------------------------
@st.cache_data
def load_dataset():
    url = "https://drive.google.com/uc?id=1FUbHoTFumMXM3kH2jfI9D-Mxao8Gl_O9"
    output = "peliculas.csv"
    gdown.download(url, output, quiet=True)
    df = pd.read_csv(output, sep=";")
    return df


# -----------------------------
# Clases de preprocesamiento
# -----------------------------
class TextContentCleaner(BaseEstimator, TransformerMixin):
    """
    Limpia los valores de texto separados por coma: hace un split por coma, convierte a minúsculas, elimina separadores, y devuelve una lista.
    """
    def __init__(self, features):
        self.features = features

    def clean_data(self, text_list):
        """Elimina espacios, convierte a minúsculas, y devuelve el string como lista, separados por la coma"""
        if isinstance(text_list, str):
            # Divide la cadena, elimina espacios, convierte a minúsculas
            return [str.lower(i.replace(" ", "")) for i in text_list.split(',')]
        # Si ya está vacío (por el transformer anterior), devuelve lista vacía
        return []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        for feature in self.features:
            # Aplica la función de limpieza
            X_copy[feature] = X_copy[feature].apply(self.clean_data)
        return X_copy

# -------------------------------
# Transformador de texto (TF-IDF)
# -------------------------------
class TfidfTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, max_features=5000):
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(max_features=self.max_features)
    def fit(self, X, y=None):
        self.vectorizer.fit(X)
        return self
    def transform(self, X):
        return self.vectorizer.transform(X)


class ColumnDropper(BaseEstimator, TransformerMixin):
  """Transformador para eliminar columnas específicas."""

  def __init__(self, columns_to_drop):
    self.columns_to_drop = columns_to_drop

  def fit(self, X, y=None):
    return self

  def transform(self, X):
  # Devuelve una copia del DataFrame sin las columnas especificadas
    return X.drop(columns=self.columns_to_drop, errors='ignore')

class SoupCreator(BaseEstimator, TransformerMixin):
    """
    Crea una 'sopa de palabras' combinando múltiples columnas de texto.
    Permite indicar:
      - qué columnas mezclar,
      - separador personalizado.
    """
    def __init__(self, columns=None, separator=" "):
        """
        columns: list
          lista de columnas que deben unirs
        separator: str
            Separador entre palabras al unir las columnas.
        """
        self.columns = columns or []
        self.separator = separator

    def _combine_row(self, row):
        parts = []
        for col in self.columns:
            value = row[col] if col in row else ""
            if isinstance(value, list):
                joined = self.separator.join(map(str, value))
            elif isinstance(value, str):
                joined = value.replace(",", self.separator)
            else:
                joined = ""
            parts.append(joined)
        return self.separator.join(filter(None, parts))

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        print(X.info())
        X_copy["soup"] = X_copy.apply(self._combine_row, axis=1)
        return X_copy


# -----------------------------
# Entrenamiento del modelo
# -----------------------------
@st.cache_resource
def train_model():
    df = load_dataset()

    text_list_cols = ["genres", "main_actors", "keywords", "directors"]
    columnas_no_usadas = ["runtime", "popularity", "vote_average", "budget", "revenue"]
    soup_columns = [
        "genres",
        "main_actors",
        "keywords",
        "directors",
        "overview",
        "production_companies",
        "title",
    ]

    preprocessor = Pipeline(
        [
            ("drop_cols", ColumnDropper(columnas_no_usadas)),
            ("clean_text", TextContentCleaner(features=text_list_cols)),
            ("soup", SoupCreator(soup_columns)),
        ]
    )

    X = preprocessor.fit_transform(df)
    tfidf = TfidfVectorizer(stop_words="english", max_features=10000)
    tfidf_matrix = tfidf.fit_transform(X["soup"])

    nn = NearestNeighbors(
        n_neighbors=11, metric="cosine", algorithm="brute", n_jobs=-1
    ).fit(tfidf_matrix)

    return df, X, tfidf, tfidf_matrix, nn


# -----------------------------
# Función pública de recomendación
# -----------------------------
def recomendar_peliculas(idx_pelicula, n=10):
    df, X, tfidf, tfidf_matrix, nn = train_model()

    distances, indices_vecinos = nn.kneighbors(tfidf_matrix[idx_pelicula], n_neighbors=n + 1)

    similitudes = 1 - distances.flatten()
    movie_indices = indices_vecinos.flatten()[1:]
    similitudes = similitudes[1:]

    df_recomendadas = (
        df.iloc[movie_indices][
            [
                "title",
                "genres",
                "directors",
                "main_actors",
                "popularity",
                "keywords",
                "overview",
                "release_date",
            ]
        ]
        .copy()
    )
    df_recomendadas["similitud"] = similitudes
    return df_recomendadas.sort_values(by="similitud", ascending=False).reset_index(
        drop=True
    )
