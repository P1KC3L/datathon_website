import json
import requests
import pandas as pd
from flask import Flask, request
from user import User

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import requests
import tensorflow as tf


from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

app = Flask(__name__)

db_source = "https://sheet.best/api/sheets/1f79da48-9ec8-4116-8347-51c5e69a1763"

df = pd.DataFrame()
mn = []


def df_to_dataset(dataframe, yout, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop(yout)
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(batch_size)
    return ds


def get_normalization_layer(name, dataset):
    # Create a Normalization layer for our feature.
    normalizer = preprocessing.Normalization(axis=None)

    # TODO
    # Prepare a Dataset that only yields our feature.
    feature_ds = dataset.map(lambda x, y: x[name])

    # Learn the statistics of the data.
    normalizer.adapt(feature_ds)

    return normalizer


def get_category_encoding_layer(name, dataset, dtype, max_tokens=None):
    # Create a StringLookup layer which will turn strings into integer indices
    if dtype == "string":
        index = preprocessing.StringLookup(max_tokens=max_tokens)
    else:
        index = preprocessing.IntegerLookup(max_tokens=max_tokens)

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])

    # Learn the set of possible values and assign them a fixed integer index.
    index.adapt(feature_ds)

    # Create a Discretization for our integer indices.
    encoder = preprocessing.CategoryEncoding(num_tokens=index.vocabulary_size())

    # Apply one-hot encoding to our indices.
    return lambda feature: encoder(index(feature))


def get_data():
    global df
    global mn
    if not (df.empty):
        return
    else:
        # Get Data from Google Sheets
        response = requests.get(db_source)
        data = response.json()

        # Create a DataFrame with data read
        df = pd.DataFrame(data)

        # Renaming columns for better processing
        df.rename(
            columns={
                "Á.liq.": "Aliquido",
                "Grupo de personal": "Grupo_de_personal",
                "CODIGO POSTAL": "Codigo_postal",
                "Motivo de la RENUNCIA": "Motivo_renuncia",
                "ReglaPHT": "Reglapht",
                "Años": "Anios",
                "Antigüedad": "Antiguedad",
                "Clave de sexo": "Sexo",
                "Lugar de nacimiento": "Lugar_de_nacimiento",
                "Edad del empleado": "Edad",
                "¿Cuanto tiempo tiene viviendo en Cd. Juarez?": "Viviendo_en_juarez",
                "Estado Civil": "Estado_civil",
                "Posición": "Posicion",
                "Tipo de Baja": "Decision_renuncia",
                "Clasificacion L. N": "Clasificacion",
            },
            inplace=True,
        )

        #  Changing data types on columns
        df["Posicion"] = df["Posicion"].astype("category")
        df["Area"] = df["Area"].astype("category")
        df["Aliquido"] = df["Aliquido"].astype("category")
        df["Grupo_de_personal"] = df["Grupo_de_personal"].astype("category")
        df["Estado_civil"] = df["Estado_civil"].astype("category")
        df["Motivo_renuncia"] = df["Motivo_renuncia"].astype("category")
        df["Banda"] = df["Banda"].astype("category")
        df["Reglapht"] = df["Reglapht"].astype("category")
        df["Sexo"] = df["Sexo"].astype("category")
        df["Estado_civil"] = df["Estado_civil"].astype("category")
        df["Decision_renuncia"] = df["Decision_renuncia"].astype("category")
        df["Viviendo_en_juarez"] = df["Viviendo_en_juarez"].astype("category")
        df["Codigo_postal"] = df["Codigo_postal"].astype("category")
        df["Hijos"] = df["Hijos"].astype("int")
        df["Edad"] = df["Edad"].astype("int")
        df["Antiguedad"] = df["Antiguedad"].astype("int")

        # Getting the indexes for all Non-voluntary Job Terminations
        index_names = df[(df["Decision_renuncia"] != "0")].index

        # drop these given row indexes from dataFrame
        df.drop(index_names, inplace=True)

        # Create target variables
        col_out = [
            "Un_anio",
            "Dos_anios",
            "Tres_anios",
            "Cuatro_anios",
            "Cinco_anios",
            "Seis_anios",
            "Siete_anios",
            "Ocho_anios",
            "Nueve_anios",
            "Diez_anios",
        ]
        for col in range(0, len(col_out)):
            df[col_out[col]] = 1
            df.loc[df["Antiguedad"] < 365 * (col + 1), col_out[col]] = 0

        # Create model
        df_model = df.copy()
        df_model.drop(
            [
                "ID",
                "Codigo_postal",
                "Baja",
                "Alta",
                "Antiguedad",
                "Lugar_de_nacimiento",
                "Viviendo_en_juarez",
                "Motivo_renuncia",
                "Antigüedad Clas",
                "Decision_renuncia",
            ],
            axis=1,
            inplace=True,
        )

        # Split data into train, validation and test
        train, test = train_test_split(df_model, test_size=0.2)
        train, val = train_test_split(train, test_size=0.2)

        # Neural Network Model
        modname = []
        for years in col_out:
            cols = col_out.copy()
            cols.remove(years)
            print(years, cols)
            trainx = train.drop(cols, axis=1)
            print(trainx.columns)
            print()
            print()
            valx = val.drop(cols, axis=1)
            print(valx.columns)
            testx = test.drop(cols, axis=1)
            batch_size = 25
            train_ds = df_to_dataset(trainx, years, batch_size=batch_size)
            val_ds = df_to_dataset(valx, years, shuffle=False, batch_size=batch_size)
            test_ds = df_to_dataset(testx, years, shuffle=False, batch_size=batch_size)
            all_inputs = []
            encoded_features = []

            # Numeric features.
            for header in ["Edad", "Hijos"]:
                numeric_col = tf.keras.Input(shape=(1,), name=header)
                normalization_layer = get_normalization_layer(header, train_ds)
                encoded_numeric_col = normalization_layer(numeric_col)
                all_inputs.append(numeric_col)
                encoded_features.append(encoded_numeric_col)

            # Categorical features encoded as string.
            categorical_cols = [
                "Posicion",
                "Area",
                "Aliquido",
                "Grupo_de_personal",
                "Banda",
                "Reglapht",
                "Sexo",
                "Clasificacion",
                "Estado_civil",
            ]
            for header in categorical_cols:
                categorical_col = tf.keras.Input(
                    shape=(1,), name=header, dtype="string"
                )
                encoding_layer = get_category_encoding_layer(
                    header, train_ds, dtype="string", max_tokens=5
                )
                encoded_categorical_col = encoding_layer(categorical_col)
                all_inputs.append(categorical_col)
                encoded_features.append(encoded_categorical_col)

            # Defining the NN layers
            all_features = tf.keras.layers.concatenate(encoded_features)
            x = tf.keras.layers.Dense(32, activation="relu")(all_features)
            x = tf.keras.layers.Dropout(0.5)(x)
            output = tf.keras.layers.Dense(1)(x)
            model = tf.keras.Model(all_inputs, output)

            # compile the model
            model.compile(
                optimizer="adam",
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=["accuracy"],
            )
            # Fit the model
            model.fit(train_ds, epochs=50, validation_data=val_ds)
            # Evaluate the model
            loss, accuracy = model.evaluate(test_ds)
            # Show the accuracy number obtained for the model
            print("\n \n \n Accuracy for ", years, "is", accuracy)
            # Creating a name to save the model
            name = years + "NN_Quit_classifier"
            model.save(name)
            modname.append(name)
            mn = modname.copy()


@app.route("/", methods=["GET"])
def go_home():
    get_data()
    return "Welcome to Datathon API"


@app.route("/getUsers", methods=["GET"])
def get_users():
    get_data()
    return df.to_json()


@app.route("/getUser/<id>", methods=["GET"])
def get_user_by_id(id):
    get_data()
    return df[df["ID"] == id].to_json()


@app.route("/getUser/prediction/<id>", methods=["GET"])
def get_user_prediction_by_id(id):
    get_data()

    #  Entering a set of parameters from an employee that stay with the Company for a few years
    anio = 0
    x_1 = []
    y_1 = []

    for names in mn:
        reloaded_model = tf.keras.models.load_model(names)

        df_user = df[df["ID"] == id]

        sample = {
            "Posicion": df_user["Posicion"].values[0],
            "Area": df_user["Area"].values[0],
            "Aliquido": df_user["Aliquido"].values[0],
            "Grupo_de_personal": df_user["Grupo_de_personal"].values[0],
            "Banda": df_user["Banda"].values[0],
            "Reglapht": df_user["Reglapht"].values[0],
            "Sexo": df_user["Sexo"].values[0],
            "Clasificacion": df_user["Clasificacion"].values[0],
            "Estado_civil": df_user["Estado_civil"].values[0],
            "Edad": df_user["Edad"].values[0],
            "Hijos": df_user["Hijos"].values[0],
        }
        input_dict = {
            name: tf.convert_to_tensor([value]) for name, value in sample.items()
        }

        anio = anio + 1

        # Creating a prediction by loading the saved model for each year
        predictions = reloaded_model.predict(input_dict)
        prob = tf.nn.sigmoid(predictions[0])

        #  Showing the prediction
        x_1.append(anio)
        y_1.append((prob.numpy()) * 100)

    return pd.DataFrame(
        list(zip(x_1, y_1)), columns=["Anios", "Probabilidad"]
    ).to_json()


@app.route("/addUser", methods=["GET", "POST"])
def add_user():
    user = User(
        request.form["position"],
        request.form["area"],
        request.form["a_liquid"],
        request.form["group"],
        request.form["cp"],
        request.form["motive"],
        request.form["fire_type"],
        request.form["band"],
        request.form["fire_date"],
        request.form["pht"],
        request.form["hire_date"],
        request.form["antiquity_years"],
        request.form["antiquity_days"],
        request.form["genre"],
        request.form["birth_place"],
        request.form["classification"],
        request.form["age"],
        request.form["living_time"],
        request.form["civil_status"],
        request.form["children"],
    )
    requests.post(
        "https://sheet.best/api/sheets/1f79da48-9ec8-4116-8347-51c5e69a1763",
        json=user.to_dict(),
    )
    print(user)
    print(user.to_dict())
    return json.dumps(user.to_dict())


if __name__ == "__main__":
    app.run(debug=False, port=8080)
