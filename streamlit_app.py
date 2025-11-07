import streamlit as st
from modelo import load_dataset, recomendar_peliculas_multiples, train_model
from grafico import generar_df_comparacion, obtener_perfil_contenido
import altair as alt

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

if "recomendadas" not in st.session_state:
    st.session_state.recomendadas = None

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
                f"{row['title']} ({row["release_date"]}) - {row['directors']}"
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
            
        poster_url = df.loc[idx_pelicula, "poster_path"]
        if isinstance(poster_url, str) and poster_url.strip() != "":
            st.image(f"https://image.tmdb.org/t/p/w300{poster_url}", width=180)
        else:
            st.write("(Sin póster disponible)")

        if st.button("+ Agregar película", key=f"add_{idx_pelicula}"):
            if idx_pelicula not in st.session_state.peliculas_idx:
                st.session_state.peliculas_idx.append(idx_pelicula)
                st.success(f"'{df.loc[idx_pelicula, 'title']}' agregada a la lista de recomendacion.")
            else:
                st.info("Esa película ya está en la lista.")

# peliculas que se van a usar para la recomendacion
if st.session_state.peliculas_idx:
    st.subheader("Películas base para recomendación:")

    # Mostrar cada película con título + poster
    cols = st.columns(len(st.session_state.peliculas_idx))

    for col, idx in zip(cols, st.session_state.peliculas_idx):
        with col:
            titulo = df.loc[idx, "title"]
            director = df.loc[idx, "directors"]
            poster_url = df.loc[idx, "poster_path"]

            st.markdown(f"**{titulo}**")
            if isinstance(poster_url, str) and poster_url.strip() != "":
                st.image(f"https://image.tmdb.org/t/p/w300{poster_url}", width=180)
            else:
                st.write("(Sin póster disponible)")

            st.caption(f"Director/es: {director}")

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
            st.session_state.recomendadas = recomendar_peliculas_multiples(
                st.session_state.peliculas_idx,
                n=n_recomendadas
            )


    if st.session_state.recomendadas is None or st.session_state.recomendadas.empty:
        st.error("No se pudieron generar recomendaciones.")
    else:
        base_titles = [df.loc[idx, "title"] for idx in st.session_state.peliculas_idx]
        st.success(f"Películas similares a **{', '.join(base_titles)}**:")
        columnas_mostradas = ["title", "release_date", "original_language", "budget", "revenue", "runtime", "vote_average", "vote_count", "genres", "similitud_promedio"]
        st.dataframe(st.session_state.recomendadas[columnas_mostradas])

        # --- selector usando el índice como value ---
        opciones = [(f"{row['title']} ({row['release_date']}) - {row["directors"]}", idx) 
                    for idx, row in st.session_state.recomendadas.iterrows()]

        seleccion_idx = st.selectbox(
            "Seleccioná una película para ver su póster:",
            options=[idx for _, idx in opciones],
            format_func=lambda x: [label for label, idx in opciones if idx == x][0]
        )

        if seleccion_idx is not None:
            fila = st.session_state.recomendadas.loc[seleccion_idx]
            poster_url = fila.get("poster_path", "")
            titulo = fila["title"]

            st.markdown(f"### {titulo}")
            if isinstance(poster_url, str) and poster_url.strip() != "":
                st.image(f"https://image.tmdb.org/t/p/w300{poster_url}", width=250)
            else:
                st.write("(Sin póster disponible)")
        
        # --- GRÁFICO INTERACTIVO ALTAR ---
        st.subheader("Comparación de Features TF-IDF")

        df_full_comparison = generar_df_comparacion(
            st.session_state.peliculas_idx,
            n_recomendaciones=n_recomendadas, n_features=15
        )

        # Selector de base
        bases = df_full_comparison['base_index'].unique()
        seleccion_base = st.selectbox(
            "Seleccioná la película base para comparar:",
            options=bases,
            format_func=lambda x: df.loc[x, 'title']
        )

        df_filtrado = df_full_comparison[df_full_comparison['base_index'] == seleccion_base]
        pelicula_base = df_filtrado['comparison_group'].unique()[0]

                # Selector de recomendada usando Altair 4.x
        selection_recom = alt.selection_single(
            fields=['comparison_group'],
            bind=alt.binding_select(
                options=[p for p in df_filtrado['comparison_group'].unique() if p != pelicula_base],
                name="Comparar con: "
            ),
            init={'comparison_group': df_filtrado['comparison_group'].unique()[1]}
        )

        chart = (
            alt.Chart(df_filtrado)
            .mark_bar()
            .encode(
                x='peso_tfidf',
                y=alt.Y('feature', sort=df_filtrado.groupby('feature')['peso_tfidf'].sum().sort_values(ascending=False).index.tolist()),
                color='comparison_group',
                tooltip=['feature', 'peso_tfidf', 'comparison_group']
            )
            # Filtrar para que siempre aparezca la base + la seleccionada
            .transform_filter(
                (alt.datum.comparison_group == pelicula_base) | selection_recom
            )
            .add_selection(selection_recom)
            .properties(title='Comparación Interactiva de Features entre Películas')
        )

        st.altair_chart(chart, use_container_width=True)
else:
    st.info("Agregá al menos una película para comenzar.")
