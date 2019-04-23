import pandas as pd
import numpy as np
from math import sin, cos, sqrt, atan2, radians

def ComputeDistance(row):
	# approximate radius of earth in km
	R = 6373.0

	lat1 = row["lat_radians"]
	lon1 = row["lon_radians"]
	lat2 = row["lat_radians1"]
	lon2 = row["lon_radians1"]

	dlon = lon2 - lon1
	dlat = lat2 - lat1

	a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
	c = 2 * atan2(sqrt(a), sqrt(1 - a))

	distance = R * c
	return distance
	
distances = pd.read_csv('data_features_clean.csv', usecols=["id","latitude","longitude"]) 	
distances["id"] = distances["id"].astype("int")
distances["lat_radians"] = distances["latitude"].apply(lambda x:radians(x))
distances["lon_radians"] = distances["longitude"].apply(lambda x:radians(x))
distances["ids"]=""
distances["distances"]=""

