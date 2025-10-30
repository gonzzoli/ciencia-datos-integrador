import streamlit as st
import os
import math
from modelo import load_dataset, recomendar_peliculas, train_model

st.set_page_config(layout="wide", page_title="Recomendador de peliculas")

st.title("Recomendador de peliculas usando Nearest Neighbors")
st.markdown("""
En esta aplicacion podras encontrar peliculas similares a alguna que te haya gustado
o parecido interesante. Para ello, simplemente debes ingresar el nombre exacto de tu pelicula y la app
te sugerira peliculas que quieras ver (cuantas vos definas).

Si hay varias peliculas con el mismo nombre que ingresaste, podras elegir la tuya en base
al director.""")

with st.spinner("Cargando dataset..."):
    df = load_dataset()
with st.spinner("Entrenando modelo..."):
    train_model()

titulo = st.text_input("Escribí el título de una película:")

if titulo:
    # Buscamos coincidencias que contengan el texto ingresado (case-insensitive)
    coincidencias = df[df["title"].str.contains(titulo, case=False, na=False)]

    if coincidencias.empty:
        st.warning("❌ No se encontró ninguna película con ese título.")
    else:
        st.write(f"Se encontraron **{len(coincidencias)}** coincidencias.")

        # Si hay varias películas con el mismo título, permitir elegir cuál
        opciones = coincidencias["title"].tolist()
        if len(coincidencias) > 1:
            opciones = [
                f"{row['title']} - {row['directors']}" for _, row in coincidencias.iterrows()
            ]
            seleccion = st.selectbox(
                "Se encontraron varias coincidencias, elegí una:",
                opciones
            )
            # Obtener el índice de la película seleccionada
            idx_peli_seleccionada = coincidencias.index[opciones.index(seleccion)]
        else:
            idx_peli_seleccionada = coincidencias.index[0]
            
        # Número de películas recomendadas
        n_recomendadas = st.number_input(
            "Cantidad de películas a recomendar:",
            min_value=1,
            max_value=50,
            value=10,
            step=1
        )

        if st.button("🔍 Recomendar"):
            with st.spinner("Buscando películas similares..."):
                recomendadas = recomendar_peliculas(
                    idx_peli_seleccionada,
                    n=n_recomendadas,
                )
            if recomendadas is None or recomendadas.empty:
                st.error("No se pudo generar recomendaciones.")
            else:
                st.success(f"Películas similares a **{df.loc[idx_peli_seleccionada]['title']}**:")
                st.dataframe(recomendadas)