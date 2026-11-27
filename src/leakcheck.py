import pandas as pd, numpy as np
D=pd.read_parquet("data/out/deals_raw.parquet"); S=pd.read_parquet("data/out/static.parquet")
c=pd.read_parquet("data/raw/customers.parquet").set_index("id"); d=pd.read_parquet("data/raw/deals.parquet").set_index("id")
X=D.merge(S,left_on="id",right_on="deal_id")
X=X[~X.instant & X.label.isin([0,1])]
X["addr"]=X.customer_id.map(c.address).notna(); X["desc"]=X.id.map(d.description).fillna("").str.len()>0
X["uid"]=X.user_id.notna(); X["zip"]=X.geo_zip.notna(); X["email"]=X.em_has==1; X["phone"]=X.ph_has==1
X["src"]=X.lead_source
for col in ["addr","desc","uid","zip","email","phone"]:
    t=X.groupby(["src","label"])[col].mean().unstack().round(2)
    t["n"]=X.groupby("src").size(); print("##",col); print(t[t.n>100].to_string())
# within won: address presence vs whether customer row updated after close
print("addr_len by label:", X.groupby("label").geo_addr_len.describe())
