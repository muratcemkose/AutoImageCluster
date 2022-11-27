"""
Created on Sun Nov  27 12:35:31 2022
@author: Murat Cem Köse
"""

import numpy as np
import pandas as pd
from datetime import datetime
from datetime import date
from exif import Image
from scipy.spatial import distance_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
import folium
from folium.plugins import MarkerCluster




def decimal_coords(coords, ref):
    """This function processes the coordiate information.
    Parameters
    ----------
    coords : List
        list of the coordinates

    ref : String
        the reference

    """
    decimal_degrees = coords[0] + coords[1] / 60 + coords[2] / 3600
    if ref == "S" or ref =='W' :
        decimal_degrees = -decimal_degrees
    return decimal_degrees

def image_coordinates_and_date(image_path):
    """This function obtains location and date of the images.
    Parameters
    ----------
    images_path : Path string
        the location of the images.

    """

    with open(image_path, 'rb') as src:
        img = Image(src)
    if img.has_exif:
        try:
            img.gps_longitude
            coords = (decimal_coords(img.gps_latitude,
                      img.gps_latitude_ref),
                      decimal_coords(img.gps_longitude,
                      img.gps_longitude_ref))
        except AttributeError:
            None
    else:
        None
        
    return({"imageTakenTime":img.datetime_original, "geolocation_lat":coords[0],"geolocation_lng":coords[1]})


def image_coordinates(image_path):
    """This function obtains only location of the images.
    Parameters
    ----------
    images_path : Path string
        the location of the images.

    """

    with open(image_path, 'rb') as src:
        img = Image(src)
    if img.has_exif:
        try:
            img.gps_longitude
            coords = (decimal_coords(img.gps_latitude,
                      img.gps_latitude_ref),
                      decimal_coords(img.gps_longitude,
                      img.gps_longitude_ref))
        except AttributeError:
            None
    else:
        None
        
    return({"imageTakenTime":np.nan, "geolocation_lat":coords[0],"geolocation_lng":coords[1]})


def image_date(image_path):
    """This function obtains only date of the images.
    Parameters
    ----------
    images_path : Path string
        the location of the images.

    """

    with open(image_path, 'rb') as src:
        img = Image(src)
    if img.has_exif:
        try:
            img.gps_longitude
            coords = (decimal_coords(img.gps_latitude,
                      img.gps_latitude_ref),
                      decimal_coords(img.gps_longitude,
                      img.gps_longitude_ref))
        except AttributeError:
            None
    else:
        None
        
    return img.datetime_original



def get_image_loc(city_database,geo_locs):
    """This function predicts the geolocation of images based on their proximity to known cities.
    Parameters
    ----------
    city_database : Data frame
        the geolocation infomation of cities.
    
    geo_locs : Data frame
        the geolocation information of the images.

    """
    euclidian_distences = pd.DataFrame(distance_matrix(city_database[["lat","lng"]].astype(float),geo_locs.iloc[:,1:3].astype(float)),index=city_database.index,columns=geo_locs.index)
    closest_loc = pd.DataFrame(euclidian_distences.idxmin().tolist(), index= geo_locs.index,columns=["city","country"])
    return closest_loc


def predict_loc(train,predict):
    """This function applies KNN classification using the geolocation data of the available images to predicts those only having the date information. To test the accuracy of the classification, it applies leave one out cross validation.
    Parameters
    ----------
    train : Data frame
        the geolocation infomation of the training dataset.
    
    predict : Data frame
        the geolocation (only date) information of the prediction dataset.

    """
    loo = LeaveOneOut()
    model_city = KNeighborsClassifier(n_neighbors=5) ### First predicting the cities
    model_city.fit(train[["imageTakenTimeTransformed"]],train["city"])
    city_scores = cross_val_score(model_city, train[["imageTakenTimeTransformed"]], train["city"], cv=loo, n_jobs=-1)

    print("Model accuracy for cities:")
    print(city_scores.sum()/len(city_scores))
    city_predictions = model_city.predict(predict[["imageTakenTimeTransformed"]])
    
    model_country = KNeighborsClassifier(n_neighbors=5)### First predicting the countries
    model_country.fit(train[["imageTakenTimeTransformed"]],train["country"])
    country_scores = cross_val_score(model_country, train[["imageTakenTimeTransformed"]], train["country"], cv=loo, n_jobs=-1)

    print("Model accuracy for countries:")
    print(country_scores.sum()/len(country_scores))
    country_predictions = model_country.predict(predict[["imageTakenTimeTransformed"]])
    
    return (city_predictions,country_predictions)

def clusters_on_map(train):
    """This function visualizes the geolocation known images on an actual map with clusters changing based on the map distance.
    Parameters
    ----------
    train : Data frame
        the geolocation infomation of the training dataset.

    """
    # Create a map object and center it to the avarage coordinates to m
    m = folium.Map(location=train[["geolocation_lat", "geolocation_lng"]].mean().to_list(), zoom_start=2)
    # if the points are too close to each other, cluster them, create a cluster overlay with MarkerCluster, add to m
    marker_cluster = MarkerCluster().add_to(m)
    # draw the markers and assign popup and hover texts
    # add the markers the the cluster layers so that they are automatically clustered
    for i,r in train.iterrows():
        location = (r["geolocation_lat"], r["geolocation_lng"])
        folium.Marker(location=location,
                          popup = i,
                          tooltip=i)\
        .add_to(marker_cluster)
    # display the map
    return m