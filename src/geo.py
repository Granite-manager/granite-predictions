"""Add zip-level geography (open uszipcode simple db, 2010 census) and distance to the shop."""
import sqlite3, numpy as np, pandas as pd

SHOP_ZIP = {1: "46202", 2: "40504", 3: "43223", 4: "36117", 7: "40228"}
zdb = pd.read_sql("select zipcode, county, state, lat, lng, population, population_density, housing_units, occupied_housing_units, "
                  "median_home_value, median_household_income, land_area_in_sqmi from simple_zipcode",
                  sqlite3.connect("data/ext/simple_db.sqlite")).set_index("zipcode")
S = pd.read_parquet("data/out/static.parquet")
z = S.geo_zip.map(lambda v: v if isinstance(v, str) else None)
G = zdb.reindex(z.values)
G.index = S.index
S["geo_lat"] = G.lat.values
S["geo_lng"] = G.lng.values
S["geo_county"] = (G.state + ":" + G.county).values
S["geo_zip_state"] = G.state.values
S["geo_pop_density"] = G.population_density.values
S["geo_population"] = G.population.values
S["geo_home_value"] = G.median_home_value.values
S["geo_income"] = G.median_household_income.values
S["geo_occupancy"] = (G.occupied_housing_units / G.housing_units).values
S["geo_land_sqmi"] = G.land_area_in_sqmi.values

def hav(lat1, lng1, lat2, lng2):
    p = np.pi / 180
    a = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lng2 - lng1) * p / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(a))

shop = {}
for cid, grp in S.groupby("company_id"):
    if cid in SHOP_ZIP and SHOP_ZIP[cid] in zdb.index:
        shop[cid] = (zdb.loc[SHOP_ZIP[cid], "lat"], zdb.loc[SHOP_ZIP[cid], "lng"])
    else:
        shop[cid] = (grp.geo_lat.median(), grp.geo_lng.median())
S["geo_dist_shop_mi"] = hav(S.geo_lat, S.geo_lng, S.company_id.map(lambda c: shop[c][0]), S.company_id.map(lambda c: shop[c][1]))
S["geo_income_rel"] = S.geo_income / S.groupby("company_id").geo_income.transform("median")
S["geo_home_value_rel"] = S.geo_home_value / S.groupby("company_id").geo_home_value.transform("median")
S.to_parquet("data/out/static.parquet")
print(S[[c for c in S.columns if c.startswith("geo_")]].notna().mean().round(2).to_string())
print(S.geo_dist_shop_mi.describe())
