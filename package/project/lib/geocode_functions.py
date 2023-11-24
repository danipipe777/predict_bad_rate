import pandas as pd
import numpy as np
from tqdm import tqdm
from arcgis.gis import GIS
from arcgis.geocoding import geocode

gis = GIS(username = "esrimo", password = "5ha6i$rE$JB*@U")

def get_coordinates(address:str):
    try:
        best_result = geocode(address)
        best_result = best_result[0]
        return(pd.Series(dict(
            lat=best_result['location']['y'],
            lon=best_result['location']['x'],
            score=best_result['score']
        )))
    except IndexError as e:
        # Cuando básciamente no encontró un resultado
        return(pd.Series(dict(
            lat=np.nan,
            lon=np.nan,
            score=np.nan
        )))
    except Exception as e:
        print(e)
        return e

def get_geocode(addresses:pd.DataFrame, verbose=0, load=False):
    addresses = addresses.copy()

    tqdm.pandas()

    addresses = (
        addresses.astype(str)
        .add(', ').sum(axis=1)
        .str.rstrip(', ')
        .str.replace('nan, ', '')
        .str.replace('\?', 'ñ')
    )

    df_coords = addresses.progress_apply(get_coordinates)
    return df_coords
