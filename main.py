#CARGAMOS LIBRERIAS
from fastapi import FastAPI
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


df_items = pd.read_csv('./items_reducido.csv')
df_reviews = pd.read_csv('./reviews_sentiment_analysis.csv')
df_genres = pd.read_csv('./games_genres.csv')
df_specs = pd.read_csv('./games_specs.csv')
df_games = pd.read_csv('./games.csv')

app = FastAPI(
    title="Steam Games Api",
    description="Esta api es creada con la intención de que se pueda obtener a traves de metodos de solicitud HTTP diferentes contenidos de nuestros dataframes de steam games",
    version="1.0.0"
    )
# Ruta principal para obtener información de la API
@app.get("/")
def read_root():
    return {"message": "Bienvenido a tu API FastAPI"}

@app.get("/playtime-genre/{genero}")
async def PlayTimeGenre(genero: str):
    """
    Devuelve el año con más horas jugadas para un género específico.
    """
    #probar borrar cosas y ver en donde deja de haber error.
    if genero not in df_genres.columns:
        no_Genre = ['El genero solicitado no esta presente en los datos']
        return no_Genre
    # Realizar un inner join entre df_genres y df_items usando 'id' como clave
    merged_df = df_genres.merge(df_items, on='item_id', how='inner')
    # Filtrar por el género deseado
    filtered_df = merged_df[merged_df[genero] == 1]
    # Realizar otro inner join con df_games usando 'id' como clave
    final_df = filtered_df.merge(df_games, on='item_id', how='inner')
    # Calcular la suma de las horas jugadas por año
    playtime_by_year = final_df.groupby(final_df['release_date'])['playtime_forever'].sum()
    # Encontrar el año con más horas jugadas
    year_with_most_playtime = playtime_by_year.idxmax().item()  # Convierte a tipo de dato nativo
    return {f"Año de lanzamiento con más horas jugadas para Género {genero}: {year_with_most_playtime}"}

@app.get("/user-for-genre/{genero}")
async def UserForGenre(genero: str):
    """
    Devuelve el usuario que ha acumulado más horas jugadas para un género dado,
    junto con una lista de la acumulación de horas jugadas por año.
    """
    if genero not in df_genres.columns:
        no_Genre = ['El genero solicitado no esta presente en los datos']
        return no_Genre
    # Realizar un inner join entre df_genres y df_items usando 'id' como clave
    merged_df = df_genres.merge(df_items, on='item_id', how='inner')
    # Filtrar por el género deseado
    filtered_df = merged_df[merged_df[genero] == 1]
    # Realizar otro inner join con df_games usando 'id' como clave
    final_df = filtered_df.merge(df_games, on='item_id', how='inner')
    # Calcular la suma de las horas jugadas por usuario y año
    user_year_playtime = final_df.groupby(['user_id', final_df['release_date']])['playtime_forever'].sum().reset_index()
    # Excluir el año 1900 de la suma de horas jugadas
    user_year_playtime = user_year_playtime[user_year_playtime['release_date'] != 1900]
    # Encontrar el usuario con más horas jugadas para el género
    user_with_most_playtime = user_year_playtime.groupby('user_id')['playtime_forever'].sum().idxmax()
    # Filtrar los datos del usuario con más horas jugadas
    user_most_playtime_data = user_year_playtime[user_year_playtime['user_id'] == user_with_most_playtime]
    # Crear una lista de acumulación de horas jugadas por año
    hours_played_by_year = [{'Año': year, 'Horas': playtime} for year, playtime in zip(user_most_playtime_data['release_date'], user_most_playtime_data['playtime_forever'])]
    # Devolver el resultado como un diccionario
    result = {
        f"Usuario con más horas jugadas para Género {genero}": user_with_most_playtime,
        "Horas jugadas": hours_played_by_year
    }
    return result

@app.get("/users-recommend/{anio}")
async def UsersRecommend(anio: int):
    """
    Devuelve el top 3 de juegos MÁS recomendados por usuarios para el año dado,
    considerando revisiones con recomendación positiva o neutral.
    """
    if anio not in df_reviews['posted'].values:
        no_anio = ['No hay reviews del año solicitado']
        return no_anio
    # Filtrar las revisiones que cumplen con los criterios de recomendación y análisis de sentimientos
    filtered_reviews = df_reviews[(df_reviews['recommend'] == True) & (df_reviews['sentiment_analysis'] <= 1)]
    # Realizar un inner join entre las revisiones y los juegos usando 'item_id' como clave
    merged_df = filtered_reviews.merge(df_games,on='item_id', how='inner')
    # Filtrar los juegos que tienen el anio deseado
    filtered_games = merged_df[merged_df['posted'] == anio]
    # Calcular la popularidad de los juegos en función de la cantidad de recomendaciones
    game_popularity = filtered_games.groupby('app_name')['recommend'].sum().reset_index()
    # Ordenar los juegos por popularidad en orden descendente
    top_games = game_popularity.sort_values(by='recommend', ascending=False).head(3)
    # Crear la lista de juegos recomendados en el formato deseado
    recommended_games = [{"Puesto {}: {}".format(index + 1, game)} for index, game in enumerate(top_games['app_name'])]
    return recommended_games

@app.get("/users-not-recommend/{anio}")
async def UsersNotRecommend(anio: int):
    """
    Devuelve el top 3 de juegos MENOS recomendados por usuarios para el año dado,
    considerando revisiones con recomendación negativa.
    """
    if anio not in df_reviews['posted'].values:
        no_anio = ['No hay reviews del año solicitado']
        return no_anio
    # Filtrar las revisiones que cumplen con los criterios de recomendación y análisis de sentimientos
    filtered_reviews = df_reviews[(df_reviews['recommend'] == False) & (df_reviews['sentiment_analysis'] == 0)]
    # Realizar un inner join entre las revisiones y los juegos usando 'item_id' como clave
    merged_df = filtered_reviews.merge(df_games, on='item_id', how='inner')
    # Filtrar los juegos que tienen el anio deseado
    filtered_games = merged_df[merged_df['posted'] == anio]
    # Calcular la popularidad de los juegos en función de la cantidad de recomendaciones
    game_popularity = filtered_games.groupby('app_name')['recommend'].sum().reset_index()
    # Ordenar los juegos por popularidad en orden descendente
    top_games = game_popularity.sort_values(by='recommend', ascending=False).head(3)
    # Crear la lista de juegos recomendados en el formato deseado
    recommended_games = [{"Puesto {}: {}".format(index + 1, game)} for index, game in enumerate(top_games['app_name'])]
    return recommended_games

@app.get("/sentiment-analysis/{anio}")
async def sentiment_analysis(anio: int):
    """ año
    Devuelve la cantidad de registros de reseñas de usuarios categorizados con
    un análisis de sentimiento para un año de lanzamiento específico.
    """
    # Filtrar los juegos del año deseado en df_games
    filtered_games = df_games[df_games['release_date'] == anio]
    # Obtener los ID de los juegos del anio deseado
    juegos_del_anio = filtered_games['item_id'].tolist()
    # Filtrar las revisiones de los juegos del anio deseado en df_reviews
    filtered_reviews = df_reviews[df_reviews['item_id'].isin(juegos_del_anio)]
    # Contar la cantidad de registros de reseñas por análisis de sentimiento
    sentiment_counts = filtered_reviews['sentiment_analysis'].value_counts()
    # Crear el diccionario de retorno en el formato deseado
    result = {
    'Negative': int(sentiment_counts.get(0, 0)),
    'Neutral': int(sentiment_counts.get(1, 0)),
    'Positive': int(sentiment_counts.get(2, 0))
}
    return result

@app.get("/recomendacion_juego/{id_producto}")
async def recomendacion_juego(id_producto:int):
    # Verificar si el ID existe en el DataFrame
    if id_producto not in df_games['item_id'].values:
        return "El item_id solicitado no pertenece a ningún juego."
    #armar el df con generos y specs
    df_gs = df_genres.merge(df_specs, on='item_id', how='inner')
    #armar el df completo
    df = df_games.merge(df_gs, on='item_id', how='inner')
    df.fillna(0, inplace=True)
    # Obtener el vector de géneros del juego de entrada
    juego_vector = df[df['item_id'] == id_producto].iloc[:, 3:].values.reshape(1, -1)
    # Calcular la similitud del coseno entre el juego de entrada y todos los demás juegos
    similarity_scores = cosine_similarity(df.iloc[:, 3:], juego_vector)
    # Obtener los índices de los juegos más similares
    similar_indices = similarity_scores.argsort(axis=0)[::-1][:5]
    # Obtener los nombres de los juegos recomendados
    recomendaciones = df.iloc[similar_indices.ravel(), :]['app_name'].values.tolist()
    return recomendaciones

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)