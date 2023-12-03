from fastapi import FastAPI
import pandas as pd
from sklearn.linear_model import LinearRegression
app = FastAPI()

@app.exception_handler(Exception)
async def exception_handler(request, exc):
    return {"error": str(exc)}

@app.get("/")
def read_root():
    try:
        df = pd.read_excel("bosch_aic_datathon.xlsx")
        return df.to_dict()
    except Exception as e:
        return {"error": str(e)}
@app.get("/predict")
def predict(x: float):
    model = LinearRegression()
    model.fit(df[["x"]], df["y"])
    y_pred = model.predict([x])
    return {"y_pred": y_pred}

@app.get("/predict")
def predict(x: float):
    model = LinearRegression()
    model.fit(df[["x"]], df["y"])
    y_pred = model.predict([x])
    return {"y_pred": y_pred}

@app.get("/predict")
def predict(x: float):
    model = LinearRegression()
    model.fit(df[["x"]], df["y"])
    y_pred = model.predict([x])
    return {"y_pred": y_pred}

@app.get("/predict")
def predict(x: float):
    model = LinearRegression()
    model.fit(df[["x"]], df["y"])
    y_pred = model.predict([x])
    return {"y_pred": y_pred}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)