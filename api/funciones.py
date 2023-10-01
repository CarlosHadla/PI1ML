#CARGAMOS LIBRERIAS
import pandas as pd
import numpy as np

df_games = pd.read_csv('./gamesNoGS.csv', parse_dates=['release_date'])
df_items = pd.read_csv('./items.csv')
df_reviews = pd.read_csv('./reviews_sentiment_analysis.csv')
df_genres = pd.read_csv('./games_genres.csv')

def PlayTime_Genre(genero: str):
    if genero not in df_genres.columns:
        return 'Invalid Genre'
    # Realizar un inner join entre df_genres y df_items usando 'id' como clave
    merged_df = df_genres.merge(df_items, left_on='id', right_on='item_id', how='inner')
    
    # Filtrar por el género deseado
    filtered_df = merged_df[merged_df[genero] == 1]
    
    # Realizar otro inner join con df_games usando 'id' como clave
    final_df = filtered_df.merge(df_games, on='id', how='inner')

    # Calcular la suma de las horas jugadas por año
    playtime_by_year = final_df.groupby(final_df['release_date'].dt.year)['playtime_forever'].sum()
    
    # Encontrar el año con más horas jugadas
    year_with_most_playtime = playtime_by_year.idxmax().item()  # Convierte a tipo de dato nativo
    
    return year_with_most_playtime


def user_for_genre(genero: str):
    if genero not in df_genres.columns:
        return None

    # Realizar un inner join entre df_genres y df_items usando 'id' como clave
    merged_df = df_genres.merge(df_items, left_on='id', right_on='item_id', how='inner')

    # Filtrar por el género deseado
    filtered_df = merged_df[merged_df[genero] == 1]

    # Realizar otro inner join con df_games usando 'id' como clave
    final_df = filtered_df.merge(df_games, on='id', how='inner')

    # Calcular la suma de las horas jugadas por usuario y año
    user_year_playtime = final_df.groupby(['user_id', final_df['release_date'].dt.year])['playtime_forever'].sum().reset_index()

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


def users_recommend(anio: int):
    if anio not in df_reviews['posted'].values:
        no_anio = ['No hay reviews del año solicitado']
        return no_anio
    # Filtrar las revisiones que cumplen con los criterios de recomendación y análisis de sentimientos
    filtered_reviews = df_reviews[(df_reviews['recommend'] == True) & (df_reviews['sentiment_analysis'] <= 1)]

    # Realizar un inner join entre las revisiones y los juegos usando 'item_id' como clave
    merged_df = filtered_reviews.merge(df_games, left_on='item_id', right_on='id', how='inner')

    # Filtrar los juegos que tienen el anio deseado
    filtered_games = merged_df[merged_df['posted'] == anio]

    # Calcular la popularidad de los juegos en función de la cantidad de recomendaciones
    game_popularity = filtered_games.groupby('app_name')['recommend'].sum().reset_index()

    # Ordenar los juegos por popularidad en orden descendente
    top_games = game_popularity.sort_values(by='recommend', ascending=False).head(3)

    # Crear la lista de juegos recomendados en el formato deseado
    recommended_games = [{"Puesto {}: {}".format(index + 1, game)} for index, game in enumerate(top_games['app_name'])]

    return recommended_games


def users_not_recommend(anio: int):
    if anio not in df_reviews['posted'].values:
        no_anio = ['No hay reviews del año solicitado']
        return no_anio
    # Filtrar las revisiones que cumplen con los criterios de recomendación y análisis de sentimientos
    filtered_reviews = df_reviews[(df_reviews['recommend'] == False) & (df_reviews['sentiment_analysis'] == 0)]

    # Realizar un inner join entre las revisiones y los juegos usando 'item_id' como clave
    merged_df = filtered_reviews.merge(df_games, left_on='item_id', right_on='id', how='inner')

    # Filtrar los juegos que tienen el anio deseado
    filtered_games = merged_df[merged_df['posted'] == anio]

    # Calcular la popularidad de los juegos en función de la cantidad de recomendaciones
    game_popularity = filtered_games.groupby('app_name')['recommend'].sum().reset_index()

    # Ordenar los juegos por popularidad en orden descendente
    top_games = game_popularity.sort_values(by='recommend', ascending=False).head(3)

    # Crear la lista de juegos recomendados en el formato deseado
    recommended_games = [{"Puesto {}: {}".format(index + 1, game)} for index, game in enumerate(top_games['app_name'])]

    return recommended_games


def sentimentAnalysis(año: int):
    # Filtrar los juegos del año deseado en df_games
    filtered_games = df_games[df_games['release_date'].dt.year == año]

    # Obtener los ID de los juegos del año deseado
    juegos_del_año = filtered_games['id'].tolist()

    # Filtrar las revisiones de los juegos del año deseado en df_reviews
    filtered_reviews = df_reviews[df_reviews['item_id'].isin(juegos_del_año)]

    # Contar la cantidad de registros de reseñas por análisis de sentimiento
    sentiment_counts = filtered_reviews['sentiment_analysis'].value_counts()

    # Crear el diccionario de retorno en el formato deseado
    result = {
    'Negative': int(sentiment_counts.get(0, 0)),
    'Neutral': int(sentiment_counts.get(1, 0)),
    'Positive': int(sentiment_counts.get(2, 0))
}

    return result