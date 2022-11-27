"""
Created on Sun Nov  27 12:35:31 2022
@author: Murat Cem Köse
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
from datetime import date
from tqdm import tqdm_notebook
import warnings
warnings.filterwarnings("ignore")
import shutil
import utils

class AutoImageClusterObject:
    def __init__(self,images_path,destination_path,image_extension=".jpg"):
        """Contructor function for SinglePython class.
        Parameters
        ----------
        images_path : Path string
            the location of the images.
        
        destination_path : Path string
            the location of destination file.
            
        image_extension : string
            the extension type of the images.
            
        """
        
        if os.path.exists(images_path):
            self.images_path = images_path
        else:
            raise Exception("Invalid path.")
        
        if os.path.exists(destination_path):
            self.destination_path = destination_path
        else:
            raise Exception("Invalid path.")
        self.image_extension = image_extension

    def read_and_categorize_images(self):
        """The function to read and categorize the images based on geolocation availability.
            
        """
        ### Obtaining the image names
        ### The city coordinates are obtanied from https://simplemaps.com
        self.city_database = pd.read_excel("./Data/worldcities.xlsx",index_col=[0,4])
        self.images = [i for i in os.listdir(self.images_path) if self.image_extension in i]
        
        ### Collecting the required metadata of the images and categorizing them based on available information
        self.valid_date_and_loc_images = []
        self.valid_date_images = []
        self.invalid_images = []
        self.geo_locs = pd.DataFrame(columns=["imageTakenTime","geolocation_lat","geolocation_lng"])
        for img in self.images:
            try:
                self.geo_locs = self.geo_locs.append(pd.DataFrame(utils.image_coordinates_and_date(self.images_path+img),index = [img]))
                self.valid_date_and_loc_images.append(img)
            except:
                try:
                    self.geo_locs.loc[img,"imageTakenTime"] = utils.image_date(self.images_path+img)
                    self.valid_date_images.append(img)
                except:
                    self.invalid_images.append(img)
        self.geo_locs["imageTakenTime"] = [datetime.strptime(i.split(" ")[0], '%Y:%m:%d') for i in self.geo_locs["imageTakenTime"]]
        self.geo_locs["year"] = [i.strftime('%Y') for i in self.geo_locs["imageTakenTime"]]
        
        print("Number of all images: "+str(len(self.images)))
        print("Number of images with location and date information: "+str(len(self.valid_date_and_loc_images)))
        print("Number of images with only date information: "+str(len(self.valid_date_images)))
        print("Number of images neither information: "+str(len(self.invalid_images)))
        
    def group_and_predict_year_and_geolocation(self):
        """The function to group the images for training and prediction data sets and predicting the geolocation of the files with missing values.
            
        """
        years = np.sort(self.geo_locs["year"].unique())
        print("Clustering procedure is initiated:")
        geo_locs_list = []
        for year in tqdm_notebook(years):
            ### Transforming the date information to the number of the date in that year
            subset_geo_locs = self.geo_locs[self.geo_locs.year==year]
            subset_geo_locs["imageTakenTimeTransformed"] = [(i-min(subset_geo_locs["imageTakenTime"])).days for i in subset_geo_locs["imageTakenTime"]]
            subset_geo_locs = subset_geo_locs.sort_values(by="imageTakenTimeTransformed")

            ### For the samples that doesnt have the location information
            ### we apply KNN classifier based on the the time taken
            ### We divide the date infor train and prediction groups
            subset_geo_locs_train = subset_geo_locs.dropna()
            subset_geo_locs_predict = subset_geo_locs[subset_geo_locs.geolocation_lat.isna()]

            ### Obtaining the city and country names of the location of the images
            closest_loc = utils.get_image_loc(self.city_database,subset_geo_locs_train)
            subset_geo_locs_train = pd.concat([subset_geo_locs_train,closest_loc],axis=1)

            print("For the year of "+year)
            ### Predicting the city and country information for missing data using KNN classifier
            city_predictions,country_predictions = utils.predict_loc(subset_geo_locs_train,subset_geo_locs_predict)
            subset_geo_locs.loc[subset_geo_locs_train.index,"city"] = subset_geo_locs_train["city"]
            subset_geo_locs.loc[subset_geo_locs_train.index,"country"] = subset_geo_locs_train["country"]
            subset_geo_locs.loc[subset_geo_locs_predict.index,"city"] = city_predictions
            subset_geo_locs.loc[subset_geo_locs_predict.index,"country"] = country_predictions
            print("")

            show_map = ""
            while show_map not in ["Yes", "No"]:
                show_map = input("Would you like to see the picture locations on a real map? Yes? No? \n")
                print("")
                if show_map not in ["Yes", "No"]:
                    print('You have entered a wrong input. Please enter one of the two possibilities.')

            if show_map == "Yes":
                display(utils.clusters_on_map(subset_geo_locs_train))

            geo_locs_list.append(subset_geo_locs)
        self.geo_locs_predicted = pd.concat(geo_locs_list,axis=0)
        
    def copy_images_to_new_loc(self):
        """The function to create missing folders and copy the images into those folders based on year, country and city information.
            
        """
        for folders in self.geo_locs_predicted.groupby(["year","country","city"]).count().index:
            year  = folders[0]
            if year not in os.listdir(self.destination_path):
                os.mkdir(self.destination_path+year+"/")
            country  = folders[1]
            if country not in os.listdir(self.destination_path+year+"/"):
                os.mkdir(self.destination_path+year+"/"+country+"/")
            city  = folders[2]
            if city not in os.listdir(self.destination_path+year+"/"+country+"/"):
                os.mkdir(self.destination_path+year+"/"+country+"/"+city+"/")
        if "Unclassified" not in os.listdir(self.destination_path):
            os.mkdir(self.destination_path+"Unclassified/")
            
        for pic in tqdm_notebook(self.images):
            if pic in self.geo_locs_predicted.index:
                pic_year = self.geo_locs_predicted.loc[pic,"year"]
                pic_country = self.geo_locs_predicted.loc[pic,"country"]
                pic_city = self.geo_locs_predicted.loc[pic,"city"]

                pic_from = self.images_path+pic
                pic_to = self.destination_path+pic_year+"/"+pic_country+"/"+pic_city+"/"+pic
            else:
                pic_from = self.images_path+pic
                pic_to = self.destination_path+"Unclassified/"+pic
            shutil.copy(pic_from,pic_to)