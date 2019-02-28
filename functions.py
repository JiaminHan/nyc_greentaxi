import os
import zipfile

import geopandas
import numpy as np
import pandas as pd
import wget
import xgboost
from geopy.distance import vincenty
from shapely.geometry import Point
from sklearn import metrics
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV, train_test_split


def geodistance(df):
    """Calculate geodistance between two points in the dataframe
        df : pandas.DataFrame or dask.DataFrame
        DataFrame containing latitudes, longitudes, and location_id columns.
     """
    localdf_geodist = df[['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']].copy()
    localdf_geodist['pickup_latitude'] = localdf_geodist['pickup_latitude'].fillna(value=0.)
    localdf_geodist['pickup_longitude'] = localdf_geodist['pickup_longitude'].fillna(value=0.)
    localdf_geodist['dropoff_latitude'] = localdf_geodist['dropoff_latitude'].fillna(value=0.)
    localdf_geodist['dropoff_longitude'] = localdf_geodist['dropoff_longitude'].fillna(value=0.)

    try:
        geo = list(localdf_geodist.apply(lambda x: vincenty((x['pickup_latitude'], x['pickup_longitude']),
                                                            (x['dropoff_latitude'], x['dropoff_longitude'])).miles,
                                         axis=1))
        return geo
    except ValueError as ve:
        print(ve)
        print(ve.stacktrace())
        geo = 0
        return geo


def download_load_shapefile():
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    if os.path.isdir('nyu_2451_36743_WGS84'):
        return
    else:
        wget.download('https://archive.nyu.edu/bitstream/2451/36743/3/nyu_2451_36743_WGS84.zip')
        zipfile.ZipFile('nyu_2451_36743_WGS84.zip', 'r').extractall()
    return


# The function is adapted from Ravi Shekhar's blog post: 'Geospatial Operations at Scale with Dask and Geopandas'
# link: https://towardsdatascience.com/geospatial-operations-at-scale-with-dask-and-geopandas-4d92d00eb7e8
def assign_taxi_zones(df, lon_var, lat_var, locid_var):
    """Joins DataFrame with Taxi Zones shapefile.
    This function takes longitude values provided by `lon_var`, and latitude
    values provided by `lat_var` in DataFrame `df`, and performs a spatial join
    with the NYC taxi_zones shapefile.
    Parameters
    ----------
    df : pandas.DataFrame or dask.DataFrame
        DataFrame containing latitudes, longitudes, and location_id columns.
    lon_var : string
        Name of column in `df` containing longitude values. Invalid values
        should be np.nan.
    lat_var : string
        Name of column in `df` containing latitude values. Invalid values
        should be np.nan
    locid_var : string
        Name of series to return.
    """

    # make a copy since we will modify lats and lons
    localdf = df[[lon_var, lat_var]].copy()

    # missing lat lon info is indicated by nan. Fill with zero
    # which is outside New York shapefile.
    localdf[lon_var] = localdf[lon_var].fillna(value=0.)
    localdf[lat_var] = localdf[lat_var].fillna(value=0.)

    shape_df = geopandas.read_file('nyu_2451_36743_WGS84/nyu_2451_36743.shp')
    shape_df.drop(['OBJECTID', "Shape_Area", "Shape_Leng", "borough", "zone"],
                  axis=1, inplace=True)
    shape_df = shape_df.to_crs({'init': 'epsg:4326'})

    try:
        local_gdf = geopandas.GeoDataFrame(
            localdf, crs={'init': 'epsg:4326'},
            geometry=[Point(xy) for xy in
                      zip(localdf[lon_var], localdf[lat_var])])

        local_gdf = geopandas.sjoin(
            local_gdf, shape_df, how='left', op='within')

        return local_gdf.LocationID.rename(locid_var)
    except ValueError as ve:
        print(ve)
        print(ve.stacktrace())
        series = localdf[lon_var]
        series = np.nan
        return series


def generate_ridge_model(X_train, y_train):
    print('Training linear regression model')
    # instantiate model
    ridge = Ridge()

    # Create param grid
    params = {
        'alpha': np.logspace(-2, 0, 4)
    }

    # Create estimator
    estimator = RandomizedSearchCV(ridge, params, n_iter=3, scoring='explained_variance')

    # Train estimator
    estimator.fit(X_train, y_train)

    return estimator


def generate_gbr_model(X_train, y_train):
    print('Training gradientboost regressor model')
    # instantiate model
    gbr = GradientBoostingRegressor()

    # Create param grid
    params = {
        'learning_rate': np.logspace(-2, 0, 4),
        'n_estimators': np.arange(20, 100, 20),
        'max_depth': np.arange(1, 15, 3),
        'max_features': ['sqrt', 'log2']
    }

    # Create estimator
    estimator = RandomizedSearchCV(gbr, params, n_iter=3, scoring='explained_variance')

    # Train estimator
    estimator.fit(X_train, y_train)

    return estimator


def generate_xgb_model(X_train, y_train):
    print('Training xgboost regressor model')
    # instantiate model
    xgb = xgboost.XGBRegressor()

    # Create param grid
    params = {
        'learning_rate': np.logspace(-2, 0, 4),
        'n_estimators': np.arange(50, 200, 50),
        'max_depth': np.arange(1, 15, 3)
    }

    # Create estimator
    estimator = RandomizedSearchCV(xgb, params, n_iter=3, scoring='explained_variance')

    # Train estimator
    estimator.fit(X_train, y_train)
    return estimator


def model_fit(observations, mapper):
    print('Begin model')
    # Reference variables
    results = list()

    # Transform dataset
    X = mapper.fit_transform(observations)

    # Extract response
    y = observations['tip_pct']

    # Split into train / test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train individual models
    model_functions = {'ridge': generate_ridge_model,
                       'gbr': generate_gbr_model,
                       'xgb': generate_xgb_model
                       }

    for (model_name, model_function) in model_functions.items():
        # Results aggregator
        local_dict = dict()
        local_dict['model_label'] = model_name

        # Create and train model
        estimator = model_function(X_train, y_train)
        local_dict['estimator'] = estimator

        # Store results
        test_preds = estimator.predict(X_test)
        local_dict['explained_variance_test'] = metrics.explained_variance_score(y_test, test_preds)
        local_dict['MSE_test'] = metrics.mean_squared_error(y_test, test_preds)
        local_dict['RMSE_test'] = np.sqrt(metrics.mean_squared_error(y_test, test_preds))

        train_preds = estimator.predict(X_train)
        local_dict['explained_variance_train'] = metrics.explained_variance_score(y_train, train_preds)
        local_dict['MSE_train'] = metrics.mean_squared_error(y_train, train_preds)
        local_dict['RMSE_train'] = np.sqrt(metrics.mean_squared_error(y_train, train_preds))

        results.append(local_dict)

    # Convert results into DataFrame
    results = pd.DataFrame(results)

    print('End model')
    return results
