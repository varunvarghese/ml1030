#Find hosts who have multiple listings across timeline

import pandas as pd
listing = pd.read_csv("../data/listings.csv")
hosts = listing.groupby(["host_id"]).count().reset_index()[["host_id","id"]].sort_values(by=["id"],ascending=False).reset_index(drop=True)
hosts_multiple_listings = hosts[hosts["id"]>1]


print(hosts_multiple_listings.head())

