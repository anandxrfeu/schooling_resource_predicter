from dotenv import load_dotenv
import os
import random

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

app = FastAPI()

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
    ibge_code = int(ibge_code)
    # lookup municipalities using IBGE code
    folder_path = "/raw_data/"
    file_name = "all_expanded_ML.csv"
    file_path = os.path.join(os.getcwd()+folder_path, file_name)
    municipality_data = pd.read_csv(file_path)
    municipality_ts = municipality_data[(municipality_data['Código_IBGE'] == ibge_code) & (municipality_data['Localização'] == localization)].sort_values(by='Ano', ascending=False).to_dict('records')
    if not municipality_ts:
        raise HTTPException(status_code=404, detail="Municipality with this configuration does not exist")
    municipality = municipality_ts[0]
    last_year = int(municipality['Ano'])
    # let range of funding values from the past
    funding = municipality_data[(municipality_data['Código_IBGE'] == ibge_code) & (municipality_data['Localização'] == localization)].sort_values(by='Ano', ascending=False)[['Ano', 'Adjusted_funding']].to_dict('records')
    adjusted_funding_values = [item['Adjusted_funding'] for item in funding]

    # make multiple prediction for all years from current to selected years
    if year - last_year > 0:
        for yr in range(last_year+1, year+1,):
            funding.append({'Ano':yr, 'Adjusted_funding' : random.choice(adjusted_funding_values)})

    funding = sorted(funding, key=lambda x: x['Ano'])
    municipality['Adjusted_funding'] = funding
    municipality['Historic_funding'] = municipality.pop('Adjusted_funding')
    return municipality
