"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  

"""

# Helper Dependencies
import numpy as np
import pandas as pd
import pickle
import json

def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.
    """
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])

    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------

    # ----------- Replace this code with your own preprocessing steps --------
    feature_vector_df['time'] = pd.to_datetime(feature_vector_df['time'])
     # day
    feature_vector_df['Day'] = feature_vector_df['time'].dt.day
    # month
    feature_vector_df['Month'] = feature_vector_df['time'].dt.month
    # year
    feature_vector_df['Year'] = feature_vector_df['time'].dt.year
    #    hour
    feature_vector_df['Start_hour'] = feature_vector_df['time'].dt.hour
    # minute
    feature_vector_df['Start_minute'] = feature_vector_df['time'].dt.minute
    # second
    feature_vector_df['Start_second'] = feature_vector_df['time'].dt.second
    # Monday is 0 and Sunday is 6
    feature_vector_df['Start_weekday'] = feature_vector_df['time'].dt.weekday
    # week of the year
    feature_vector_df['Start_week_of_year'] = feature_vector_df['time'].dt.week
    # duration
    feature_vector_df['Valencia_wind_deg']=feature_vector_df['Valencia_wind_deg'].str.extract('(\d+)')
    feature_vector_df['Valencia_wind_deg'] = pd.to_numeric(feature_vector_df['Valencia_wind_deg'])
    feature_vector_df['Seville_pressure']=feature_vector_df['Seville_pressure'].str.extract('(\d+)')
    feature_vector_df['Seville_pressure'] = pd.to_numeric(feature_vector_df['Seville_pressure'])
    feature_vector_df = feature_vector_df.fillna(value=feature_vector_df['Valencia_pressure'].mean())
    #removing the unnamed column
    feature_vector_df.drop('Unnamed: 0', inplace =True, axis=1)

    #dropping column that have only zero values
    feature_vector_df.drop(columns =['Valencia_snow_3h', 'Bilbao_snow_3h', 'Seville_rain_3h', 'Barcelona_rain_3h'], inplace= True)
    new_feature_optimal =['Madrid_wind_speed', 'Valencia_wind_deg', 'Valencia_wind_speed',
       'Seville_humidity', 'Madrid_humidity', 'Bilbao_wind_speed',
       'Bilbao_wind_deg', 'Barcelona_wind_speed', 'Barcelona_wind_deg',
       'Seville_wind_speed', 'Seville_pressure', 'Barcelona_pressure',
       'Barcelona_weather_id', 'Bilbao_pressure', 'Valencia_pressure',
       'Seville_temp_max', 'Madrid_pressure', 'Bilbao_weather_id',
       'Valencia_humidity', 'Valencia_temp_min', 'Madrid_temp_max',
       'Barcelona_temp', 'Bilbao_temp_min', 'Barcelona_temp_min',
       'Bilbao_temp_max', 'Seville_temp_min', 'Madrid_temp_min', 'Year',
       'Month', 'Day', 'Start_hour', 'Start_weekday', 'Start_week_of_year']
    X= feature_vector_df[new_feature_optimal]
    y=feature_vector_df['load_shortfall_3h']
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_standardise = pd.DataFrame(X_scaled,columns=X.columns)
    feature_vector_df = pd.concat([X_standardise, y], axis = 1)

    # ------------------------------------------------------------------------

    return feature_vector_df

def load_model(path_to_model:str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    return pickle.load(open(path_to_model, 'rb'))


""" You may use this section (above the make_prediction function) of the python script to implement 
    any auxiliary functions required to process your model's artifacts.
"""

def make_prediction(data, RF_model):
    """Prepare request data for model prediction.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    prediction = RF_model.predict(prep_data)
    # Format as list for output standardisation.
    return prediction[0].tolist()
