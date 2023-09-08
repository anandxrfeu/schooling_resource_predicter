from dotenv import load_dotenv
import os
import random

import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

app = FastAPI()

def load_saved_model():
    folder_path = "/models/"
    file_name = "model.h5"
    file_path = os.path.join(os.getcwd()+folder_path, file_name)
    return load_model(file_path)

def ml_preproc():
    folder_path = "/raw_data/"
    file_name = "all_urban_expanded_ML.csv"
    file_path = os.path.join(os.getcwd()+folder_path, file_name)
    df_urban = pd.read_csv(file_path)
    # Sort by 'Ano' and 'Código_IBGE' and group by 'Código_IBGE'
    df_urban_sorted = df_urban.sort_values(by=['Código_IBGE', 'Ano'])
    grouped_urban = [group for _, group in df_urban_sorted.groupby('Código_IBGE')]

    # Filter each group to include data from 2012 to 2020 and create sequences
    def filter_and_create_sequences(grouped_data):
        sequences = []
        targets = []
        for group in grouped_data:
            filtered_group = group[(group['Ano'] >= 2012) & (group['Ano'] <= 2020)]
            if len(filtered_group) == 9:
                sequence = filtered_group.drop(columns=['Ano', 'Código_IBGE', 'Adjusted_funding']).values
                target_values = filtered_group['Adjusted_funding'].values
                sequences.append(sequence)
                targets.append(target_values)
        return np.array(sequences), np.array(targets)


    # Create sequences and targets for urban datasets
    array_urban, y_urban = filter_and_create_sequences(grouped_urban)

    # Split data into training and testing sets
    def split_data(features, targets):
        X_train = features[:, :6, :]
        X_test = features[:, 6:, :]
        y_train = targets[:, 6:]
        y_test = targets[:, 6:]
        return X_train, X_test, y_train, y_test

    # Split the data for both urban datasets
    X_train_urban, X_test_urban, y_train_urban, y_test_urban = split_data(array_urban, y_urban)

    return y_train_urban

app.state.model = load_saved_model()
app.state.y_train_urban = ml_preproc()


# Load .env file
load_dotenv()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def root():
    #return dict(greeting="Hello World!!")
    return {"message": f"Hello {os.environ.get('SECRET_WORD')}!"}


@app.get("/predict-funding/{ibge_code}")
async def predict_funding(ibge_code, year: int = 2023, localization: str = 'Urbana', passing_rate: float = 90):
    # Convert IBGE code to integer
    ibge_code = int(ibge_code)

    # lookup municipalities using IBGE code
    folder_path = "/raw_data/"
    file_name = "all_expanded_ML.csv"
    file_path = os.path.join(os.getcwd()+folder_path, file_name)
    municipality_data = pd.read_csv(file_path)
    municipality_ts = municipality_data[(municipality_data['Código_IBGE'] == ibge_code) & (municipality_data['Localização'] == localization)].sort_values(by='Ano', ascending=False).to_dict('records')
    if not municipality_ts:
        raise HTTPException(status_code=404, detail="Municipality with this configuration does not exist")

    # Get the most recent record for the municipality
    municipality = municipality_ts[0]
    print(municipality)
    last_year = int(municipality['Ano'])

    # Extract the funding values from the past for the same municipality and localization
    funding = municipality_data[(municipality_data['Código_IBGE'] == ibge_code) & (municipality_data['Localização'] == localization)].sort_values(by='Ano', ascending=False)[['Ano', 'Adjusted_funding']].to_dict('records')
    adjusted_funding_values = [item['Adjusted_funding'] for item in funding]

    # Make multiple prediction for all years from current to selected years
    if year - last_year > 0:
        for yr in range(last_year+1, year+1,):
            funding.append({'Ano':yr, 'Adjusted_funding' : random.choice(adjusted_funding_values)})

    funding = sorted(funding, key=lambda x: x['Ano'])
    # Update the municipality dictionary with the newly generated funding data
    municipality['Adjusted_funding'] = funding
    municipality['Historic_funding'] = municipality.pop('Adjusted_funding')

    return municipality


@app.get("/predict-funding/v2/{ibge_code}")
async def new_end_point(ibge_code, year: int = 2023, localization: str = 'Urbana', passing_rate: float = 90):

    # to be removed
    localization = 'Urbana'

    # Convert IBGE code to integer
    ibge_code = int(ibge_code)

    # lookup municipalities using IBGE code
    folder_path = "/raw_data/"
    file_name = "all_expanded_ML.csv"
    file_path = os.path.join(os.getcwd()+folder_path, file_name)
    municipality_data = pd.read_csv(file_path)
    #print("municipality_data b4>> ", municipality_data.head(3))

    municipality_data = municipality_data[(municipality_data['Código_IBGE'] == ibge_code) &  (municipality_data['Localização'] == localization)].sort_values(by='Ano', ascending=False)

    #print("municipality_data afr >> ", municipality_data.head(3))

    if municipality_data.empty:
        raise HTTPException(status_code=404, detail="Municipality with this configuration does not exist")





    ## Make prediction ##

    # # Prepare data
    municipality_ML = municipality_data.copy()[['Ano', 'Código_IBGE', 'Aprovação', 'Reprovação', 'Abandono',
       'Matrículas', 'Docentes', 'Estabelecimentos', 'Turmas', 'PIB',
       'Poverty_%', 'Unemployed_%', 'Acesso_a_internet_%',
       'Adjusted_population', 'Adjusted_funding']]

    #

    # Scale the features
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    sample_features_scaled = feature_scaler.fit_transform(municipality_ML.drop(columns=['Ano', 'Código_IBGE', 'Adjusted_funding']))

    # Load model
    new_model = app.state.model

    # Get preprocessed target
    y_train_urban = app.state.y_train_urban

    # Make predictions using the model
    predictions_scaled = new_model.predict(sample_features_scaled[-3:,:].reshape(1, 3, -1))

    # Inverse transform the scaled predictions to the original scale
    target_scaler = MinMaxScaler().fit(y_train_urban)  # Assuming y_train_urban is your training target data

    # The predictions for the next three years are now stored in predictions_original_scale
    predictions_original_scale = target_scaler.inverse_transform(predictions_scaled)

    ## Combine Prediction data with Original data ##


    # Create a dictionary with the first row of the dataset, using column names as keys
    municipality_data_2020 = municipality_data.iloc[0].to_dict()

    # Create a dictionary with 'Ano' (Year) as the key and 'Adjusted_funding' as the value
    adjusted_funding_dict = dict(zip(municipality_data['Ano'], municipality_data['Adjusted_funding']))

    # Given list of lists with funding information for years starting from 2021 and onwards
    #funding_list_of_lists = [[12102227.0, 13575426.0, 13404874.0]]
    funding_list_of_lists = predictions_original_scale.flatten().tolist()

    # Generate years for the new funding information, starting from 2021
    new_funding_years = list(range(2021, 2021 + len(funding_list_of_lists)))

    # Create a new dictionary with the new funding information
    new_funding_dict = dict(zip(new_funding_years, funding_list_of_lists))

    # Update the 'adjusted_funding_dict' with the new funding information
    adjusted_funding_dict.update(new_funding_dict)

    # Add the 'adjusted_funding_dict' as a value in the 'first_row_dict' under the key 'Historical_funding'
    municipality_data_2020['Historic_funding'] = adjusted_funding_dict

    # Transform the dictionary into a list of dictionaries with 'Ano' and 'Adjusted_funding' keys
    transformed_historic_funding = [{"Ano": year, "Adjusted_funding": funding} for year, funding in municipality_data_2020['Historic_funding'].items()]
    # Sort the transformed list of dictionaries by the 'Ano' key
    sorted_transformed_historic_funding = sorted(transformed_historic_funding, key=lambda x: x['Ano'])
    municipality_data_2020['Historic_funding'] = sorted_transformed_historic_funding

    # Display the transformed 'Historic_funding'
    transformed_historic_funding

    # Delete the
    del municipality_data_2020["Adjusted_funding"]

    return municipality_data_2020
