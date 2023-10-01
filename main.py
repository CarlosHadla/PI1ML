#CARGAMOS LIBRERIAS
from fastapi import FastAPI
from api.funciones import PlayTime_Genre, user_for_genre, users_recommend, users_not_recommend, sentimentAnalysis
import pandas as pd

app = FastAPI(
    title="Steam Games Api",
    description="Esta api es creada con la intención de que se pueda obtener a traves de metodos de solicitud HTTP diferentes contenidos de nuestros dataframes de steam games",
    version="1.0.0"
    )

@app.get("/playtime-genre/{genero}")
async def PlayTimeGenre(genero: str):
    return {"year_with_most_playtime": PlayTime_Genre(genero)}

@app.get("/user-for-genre/{genero}")
async def UserForGenre(genero: str):
    return user_for_genre(genero)

@app.get("/users-recommend/{anio}")
async def UsersRecommend(anio: int):
    return users_recommend(anio)

@app.get("/users-not-recommend/{anio}")
async def UsersNotRecommend(anio: int):
    return users_not_recommend(anio)

@app.get("/sentiment-analysis/{anio}")
async def sentiment_analysis(anio: int):
    return sentimentAnalysis(anio)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)