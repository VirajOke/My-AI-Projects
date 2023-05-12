# Databricks notebook source
# Setup
!pip install /Workspace/Repos/Production/dsai_databricks_helpers
import dsai_databricks_helpers as dsai
import pandas as pd
import numpy as np

# COMMAND ----------

epet = spark.sql(
    """SELECT * from vizhelper.epet where lower (fund_type) = ('salary')"""
).toPandas()  # only the salary records are important for LoE
bits = spark.sql(
    """SELECT BR_NMBR, BR_TITLE, BR_SHORT_TITLE, PRIORITY_ID, REQMT_OVRVW, LEAD_SERVICE_ID from edr.demand_br_bits_items"""
).toPandas()
bits["BR_NMBR"] = bits["BR_NMBR"].astype("int")
items_prod = spark.sql(
    "select distinct Service_ID, Service_DESC_EN as SERVICE_DESC_EN from edr.demand_br_bits_item_to_product"
).toPandas()  # bringing service data from a different table
bits = pd.merge(
    bits, items_prod, how="left", left_on=["LEAD_SERVICE_ID"], right_on=["Service_ID"]
)
epet["brnumber"] = (
    epet["brnumber"].apply(pd.to_numeric, errors="coerce").fillna(0, downcast="infer")
)
epet["onetime_total"] = (
    epet["onetime_total"]
    .apply(pd.to_numeric, errors="coerce")
    .fillna(0, downcast="infer")
)  # this column has a weird type, need to fix it

# COMMAND ----------

epet = epet.drop_duplicates(keep="first")
epet_reduced = epet[
    [
        "fiscalyear",
        "customer",
        "brnumber",
        "service",
        "fa",
        "cost_centre",
        "cost",
        "onetime_total",
        "item_description",
    ]
].copy()  # picking important columns
# epet_reduced

# COMMAND ----------

epet_reduced

# COMMAND ----------

# joining both tables using br number
epet_bits = pd.merge(
    epet_reduced, bits, how="inner", left_on=["brnumber"], right_on=["BR_NMBR"]
)

# COMMAND ----------

# epet_bits #looking at the table after joining

# COMMAND ----------

# MAGIC %md We can see that we have multiple services for the same BR for epet. I brought in the service data from the items_prod table and that may be more consistent

# COMMAND ----------

# DBTITLE 1,Group by BRs to sum cost columns
epet_bits_final1 = (
    epet_bits.groupby(
        [
            "BR_NMBR",
            "fiscalyear",
            "customer",
            "REQMT_OVRVW",
            "SERVICE_DESC_EN",
            "PRIORITY_ID",
        ]
    )["cost"]
    .sum()
    .reset_index()
)
epet_bits_final2 = epet_bits.groupby(["BR_NMBR"])["onetime_total"].sum().reset_index()
epet_bits_final = pd.merge(
    epet_bits_final1,
    epet_bits_final2,
    how="inner",
    left_on=["BR_NMBR"],
    right_on=["BR_NMBR"],
)
epet_bits_final = epet_bits_final.round({"cost": 2})
epet_bits_final = epet_bits_final.round({"onetime_total": 2})

# COMMAND ----------

epet_bits_final

# COMMAND ----------

# MAGIC %md
# MAGIC Only 48 records meet the requirement for computing level of effort, this is not enough data points to train a model to predict the LoE

# COMMAND ----------

print(epet_bits_final.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC [Resource](https://medium.com/analytics-vidhya/multiclass-text-classification-using-deep-learning-f25b4b1010e5)
# MAGIC 
# MAGIC We will try to use ML to classify regardless using the methods proposed in the above link, but let's try to do that for predicting the service

# COMMAND ----------

# MAGIC %md
# MAGIC We do not need the epet table to build a model to predict the service, we can do that using the bits table directly

# COMMAND ----------

bits_service = spark.sql(
    """SELECT BR_NMBR, REQMT_OVRVW as Description, LEAD_SERVICE_ID from edr.demand_br_bits_items"""
).toPandas()
bits_service["BR_NMBR"] = bits_service["BR_NMBR"].astype("int")
items_prod = spark.sql(
    "select distinct Service_ID, Service_DESC_EN as Service from edr.demand_br_bits_item_to_product"
).toPandas()  # bringing service data from a different table
bits_service = pd.merge(
    bits_service,
    items_prod,
    how="left",
    left_on=["LEAD_SERVICE_ID"],
    right_on=["Service_ID"],
)
bits_service.drop(["LEAD_SERVICE_ID", "Service_ID"], axis=1, inplace=True)

# COMMAND ----------

bits_service  # we can use this table to train a model to use the description to predict the service

# COMMAND ----------

# Converting the service to a number to create code table
numeric_service = bits_service["Service"].unique()
numeric_service = pd.DataFrame(numeric_service, columns=["Service"])
numeric_service["service_num"] = numeric_service.index + 1
numeric_service #visualize the code table

# COMMAND ----------

#join code table to original  to original table
bits_service_final = pd.merge(
    bits_service, numeric_service, how="left", left_on=["Service"], right_on=["Service"]
)
bits_service_final

# COMMAND ----------

#writing the final table to the mart (this table is to be used for ML)
dsai.create_overwrite_table(bits_service_final, "bits", "epet_bits")
