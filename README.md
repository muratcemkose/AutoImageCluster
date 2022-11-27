# AutoImageCluster
- A python library to cluster your images easily based on geolocation.

- Its a pet project of mine to group the images based on geolocation. Although there are applications for this purpose, onces you want to move the files to another platform, the groups disappear, since they use referencing. 

- Using this Python library, one can group the images based on their geolocation. 
- Those images lacking the geolocation info but having date information, algorithm runs a KNN classfier and annotates them based on the geolocation of the nearest 5 time points. 
- KNN classification is applied per year and the performance of each KNN classifier is assessed using leave-one-out cross validation.
- At the end, the custered images are copied to a destination in the year/country/city/ format.
