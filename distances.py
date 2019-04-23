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
	
listing = pd.read_csv("data_features_clean.csv")
	

distances = listing[["id","latitude","longitude"]]
distances["id"] = distances["id"].astype("int")
distances["lat_radians"] = distances["latitude"].apply(lambda x:radians(x))
distances["lon_radians"] = distances["longitude"].apply(lambda x:radians(x))
distances["ids"]=""
distances["distances"]=""

#Take a copy of the distances dataframe
distances_copy = distances.copy()

for i in distances.index:
	id = distances.iloc[i]["id"]
	lat_radians = distances.iloc[i]["lat_radians"]
	lon_radians = distances.iloc[i]["lon_radians"]

	distances_copy["id1"] = id
	distances_copy["lat_radians1"] = lat_radians
	distances_copy["lon_radians1"] = lon_radians
	distances_copy["distance"] = distances_copy.apply(ComputeDistance, axis=1)
	ids = str(list(distances_copy[distances_copy["id"] != distances_copy["id1"]].sort_values(by="distance", ascending=True).head(20)["id"])).replace("[","").replace("]","")
	dists = str(list(distances_copy[distances_copy["id"] != distances_copy["id1"]].sort_values(by="distance",ascending=True).head(20)["distance"])).replace("[","").replace("]","")
	
	distances.loc[i, "ids"] = ids
	distances.loc[i, "distances"] = dists
	

distances.to_csv("distances.csv",index=False, header=True)





