import pandas as pd
import numpy as np

df = pd.read_csv("../data/data_features_clean.csv")

#retain only selected columns
df = df[["id","host_id", "latitude","longitude"]]
print(df.head())

#delete duplicate rows
df_unique = df.drop_duplicates(["host_id","latitude","longitude"])[["host_id","latitude","longitude"]]

#find the combinations of latitude,longitude with multiple host_ids
host_count = pd.pivot_table(df_unique[["host_id","latitude","longitude"]], index=["latitude","longitude"], aggfunc=np.count_nonzero).reset_index()
print("len(df): ", len(df))
print("len(df_unique): ", len(df_unique))
print("len(host_count): ", len(host_count))

#convert the coordinates to float if they are not float already
df_unique["latitude"] = df_unique["latitude"].astype("float64")
df_unique["longitude"] = df_unique["longitude"].astype("float64")
host_count["latitude"] = host_count["latitude"].astype("float64")
host_count["longitude"] = host_count["longitude"].astype("float64")


#rename the columns
host_count.columns = ["latitude", "longitude", "count"]

#find the latitude and longitude combinations which have multiple host_ids mapped to them
multiple_hosts_per_latlon = host_count[host_count["count"]>1].reset_index(drop=True)
print(multiple_hosts_per_latlon)

#print the rows from df_unique for the combination of lat/lon which have multiple host_ids mapped to them
for i in multiple_hosts_per_latlon.index: 
	lat = multiple_hosts_per_latlon.iloc[i]["latitude"] 
	lon = multiple_hosts_per_latlon.iloc[i]["longitude"] 
	print(df_unique[(df_unique["latitude"]==lat) & (df_unique["longitude"]==lon)]) 
	print("") 
	
