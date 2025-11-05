import streamlit as st
from modelo import load_dataset, recomendar_peliculas_multiples, train_model

st.set_page_config(layout="wide", page_title="Recomendador de Películas")

st.title("Recomendador de Películas")
st.markdown("""
En esta aplicacion podes buscar una o mas películas por su título, y si hay varias coincidencias podes 
identificar la tuya segun su director.
Con todas las peliculas que agregues, vamos a recomendarte todas las peliculas que vos quieras que 
pensamos que te gustarian ver.
""")

# Carga del dataset y modelo
with st.spinner("Cargando dataset..."):
    df = load_dataset()
with st.spinner("Entrenando modelo..."):
    train_model()

# Estado para guardar índices seleccionados
if "peliculas_idx" not in st.session_state:
    st.session_state.peliculas_idx = []

st.subheader("Buscar película")
titulo = st.text_input("Escribí el título de una película:")
# Se va buscando pelicula por pelicula, y cada vez que agregue una al peliculas_idx
# va apareciendo en el listado de abajo
if titulo:
    # resulta en un dataframe
    coincidencias = df[df["title"].str.contains(titulo, case=False, na=False)]

    if coincidencias.empty:
        st.warning("❌ No se encontró ninguna película con ese título.")
    else:
        if len(coincidencias) > 1:
            opciones = [
                f"{row['title']} - {row['directors']}"
                for _, row in coincidencias.iterrows()
            ]
            seleccion = st.selectbox(
                "Se encontraron varias coincidencias, elegí una:",
                opciones,
                key=f"select_{titulo}"
            )
            idx_pelicula = coincidencias.index[opciones.index(seleccion)]
        else:
            idx_pelicula = coincidencias.index[0]
            st.info(f"Se seleccionó: {df.loc[idx_pelicula, 'title']}")

        if st.button("+ Agregar película", key=f"add_{idx_pelicula}"):
            if idx_pelicula not in st.session_state.peliculas_idx:
                st.session_state.peliculas_idx.append(idx_pelicula)
                st.success(f"'{df.loc[idx_pelicula, 'title']}' agregada a la lista base.")
            else:
                st.info("Esa película ya está en la lista.")

# peliculas que se van a usar para la recomendacion
if st.session_state.peliculas_idx:
    st.subheader("Películas base para recomendación:")
    for idx in st.session_state.peliculas_idx:
        st.write(f"- {df.loc[idx, 'title']} ({df.loc[idx, 'directors']})")

    if st.button("X - Limpiar lista"):
        st.session_state.peliculas_idx = []
        st.info("Lista vaciada.")

    n_recomendadas = st.number_input(
        "Cantidad de películas a recomendar:",
        min_value=1,
        max_value=50,
        value=10,
        step=1
    )

    if st.button("Generar recomendaciones"):
        with st.spinner("Buscando películas similares..."):
            recomendadas = recomendar_peliculas_multiples(
                st.session_state.peliculas_idx,
                n=n_recomendadas
            )

        if recomendadas is None or recomendadas.empty:
            st.error("No se pudieron generar recomendaciones.")
        else:
            base_titles = [df.loc[idx, "title"] for idx in st.session_state.peliculas_idx]
            st.success(f"Películas similares a **{', '.join(base_titles)}**:")
            st.dataframe(recomendadas)
else:
    st.info("Agregá al menos una película para comenzar.")
