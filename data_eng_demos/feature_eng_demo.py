print("Importing libraries ... ")
import os
import pandas as pd
import psutil
import numpy as np
import random
print("All modules are ready")

# Setting up the system
def settings():
    # Display python version and operational system-specif parameters 
    import sys
    print("Python version: ", sys.version, "\n")

    # Check the number of cores and memory usage
    import multiprocessing as mp
    num_cores = mp.cpu_count()
    import psutil
    print("This kernel has ", num_cores, "cores and memory usage of:", psutil.virtual_memory(), "\n")

    # Expands the visualization of a matrix
    pd.set_option("display.max_columns", 500)
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.width", 500)
    
    #Checking the directory we are working on
    import os
    print("File directory", os.getcwd())

settings()

# COMMAND ----------

print("Conversion toPandas() starting...")

hawkeye_metrics = spark.sql('''SELECT
    A.PROBE_TEST_SID as Test_ID,
    METRIC_TYPE_DESC_EN as Metric,
    THRESHOLD as Threshold,
    VALUE as Value,
    METRIC_PAIR_NAME_DESC_EN as Application
    --METRIC_STATUS_DESC_EN
FROM
    nss_hawkeye.f_probe_test A
LEFT JOIN
    nss_hawkeye.f_probe_metric B ON A.PROBE_TEST_SID = B.PROBE_TEST_SID
LEFT JOIN
    nss_hawkeye.d_metric_type C ON C.METRIC_TYPE_SID = B.METRIC_TYPE_SID
LEFT JOIN
    nss_hawkeye.d_metric_pair_name D ON D.METRIC_PAIR_NAME_SID = B.PAIR_NAME_SID
LEFT JOIN
    nss_hawkeye.d_metric_status E ON E.METRIC_STATUS_SID = B.STATUS_SID
where
(lower (METRIC_TYPE_DESC_EN) like "%jitter%" or lower (METRIC_TYPE_DESC_EN) like "%%delay%")
--LIMIT 1000    
''').toPandas()

print("Conversion .toPandas() done!")

# COMMAND ----------

hawkeye = hawkeye_metrics
# Renaming columns
hawkeye.columns = ['Test_ID', 'Metric', 'Threshold', 'Value', 'Application']

# reordering columns
column_order = ['Test_ID', 'Metric', 'Application', 'Threshold', 'Value']
hawkeye = hawkeye[column_order]

hawkeye.head() 

# COMMAND ----------

# Transforming all the 0 values to 0.01
hawkeye.loc[:, "Value"] = pd.to_numeric(hawkeye["Value"], errors='coerce')
hawkeye.loc[hawkeye["Value"] == 0, "Value"] = 0.01

# COMMAND ----------

# making a copy of hawkeye 

df1 = hawkeye.copy() 

df1.head()

# COMMAND ----------

# Splitting the dataframe into two new dataframe with Delay and Jitter as metric 

df_delay = df1[df1["Metric"] == "Delay (ms)"][["Test_ID", "Metric", "Value", "Application", "Threshold"]].replace(0, pd.NA).reset_index(drop=True)
df_jitter = df1[df1["Metric"] == "Jitter (ms)"][["Test_ID", "Metric", "Value", "Application", "Threshold"]].replace(0, pd.NA).reset_index(drop=True)

df_delay[:15]

#combined_df = dummy_df_even.combine_first(dummy_df_odd)


#combined_df[:15]

# COMMAND ----------

# Pivot the table using pivot_table with an aggregation function
df2 = df_delay.pivot_table(index="Test_ID", columns="Metric", values="Value", aggfunc=list).reset_index()

# Separate all values on their own rows using explode()
df2 = df2.explode("Delay (ms)").reset_index(drop=True)
df2 = df2.rename(columns={"Delay (ms)": "Delay"})

# Add the jitter column with null values 
df2["Jitter"] = np.nan


df2[:15]

# COMMAND ----------

# Pivot the table using pivot_table with an aggregation function
df3 = df_jitter.pivot_table(index="Test_ID", columns="Metric", values="Value", aggfunc=list).reset_index()

# Separate all values on their own rows using explode()
df3 = df3.explode("Jitter (ms)").reset_index(drop=True)
df3 = df3.rename(columns={"Jitter (ms)": "Jitter"})

# Add the Delay column with null values 
df3.insert(1, "Delay", np.nan)

df3.head()

# COMMAND ----------

# use combine_first() to merge the 2 df together

df4 = df2.combine_first(df3)

df4[:15]

# COMMAND ----------

# df5 = Threshold/Delay 
# df6 = Threshold/Jitter

# COMMAND ----------

# Create threshold/metric columns 

# Pivot the table using pivot_table with an aggregation function
df5 = df_delay.pivot_table(index="Test_ID", columns="Metric", values="Threshold", aggfunc=list).reset_index()

# Separate all values on their own rows using explode()
df5 = df5.explode("Delay (ms)").reset_index(drop=True)
df5 = df5.rename(columns={"Delay (ms)": "Threshold/Delay"})

# Add the Threshold/Jitter column with null values 
df5["Threshold/Jitter"] = np.nan

df5[:15]

# COMMAND ----------

# Create threshold/metric columns 

# Pivot the table using pivot_table with an aggregation function
df6 = df_jitter.pivot_table(index="Test_ID", columns="Metric", values="Threshold", aggfunc=list).reset_index()

# Separate all values on their own rows using explode()
df6 = df6.explode("Jitter (ms)").reset_index(drop=True)
df6 = df6.rename(columns={"Jitter (ms)": "Threshold/Jitter"})

# Add the Delay column with null values 
df6.insert(1, "Threshold/Delay", np.nan)

df6[:15]

# COMMAND ----------

# use combine_first() to merge the 2 df together

df7 = df5.combine_first(df6)

df7[:15]

# COMMAND ----------

# combine df4 with df7

df8 = df4.combine_first(df7)

# reorder columns
df8 = df8[["Test_ID", "Delay", "Jitter", "Threshold/Delay", "Threshold/Jitter"]]

df8[:15]


# COMMAND ----------

# Create Ratio/Delay and Ratio/Jitter columns 

df8["Ratio/Delay"] = df8["Delay"]/df8["Threshold/Delay"]
df8["Ratio/Jitter"] = df8["Jitter"]/df8["Threshold/Jitter"]

df8[:15]

# COMMAND ----------

# Adding a max_ratio(Delay, Jitter) column

# Calculate the max ratio between Delay and Jitter
df8['Max_Ratio(Delay/Jitter)'] = df8[['Ratio/Delay', 'Ratio/Jitter']].max(axis=1)

df8[:15]

# COMMAND ----------

# Adding the quality of experience column

# Defining all the conditions inside a function
def condition(x):
    if x< 0.2:
        return "very good"
    elif x>=0.2 and x<0.4:
        return "good"
    elif x>=0.4 and x<0.6:
        return "fair"
    elif x>=0.6 and x<0.8:
        return "bad"
    elif x>=0.8:
        return "very bad"
    else:
        return "not valid"
 
# Applying the conditions
df8['Quality of experience'] = df8['Max_Ratio(Delay/Jitter)'].apply(condition)

df8[:15]

# COMMAND ----------

# Create get_dummies df

dummy_df = pd.get_dummies(df1["Application"])

dummy_df[:15]

# COMMAND ----------

# combine the two df together 

df9 = df8.combine_first(dummy_df)

df9.head()

# COMMAND ----------

# Reordering the columns (putting max_ratio and Quality of Experience to the end)

df9 = df9[["Test_ID", "Delay", "Jitter", "Ratio/Delay", "Ratio/Jitter", "Audio RTP from->to 20 kbps", "Audio RTP to->from 20 kbps", "Video RTP from->to 150 kbps", "KPI from->to", "KPI to->from", "Max_Ratio(Delay/Jitter)", "Quality of experience"]]

df9[:15]

# COMMAND ----------

# # Create get_dummies df 

# # Concatenate "Application" and "Metric" columns
# df1["Application/Metric"] = df1["Application"] + "/" + df1["Metric"]

# df1[:15]

# COMMAND ----------

# # One hot encoding
# dummy_df = pd.get_dummies(df1["Application/Metric"])

# dummy_df[:15]

# COMMAND ----------

# # Add the ID column to the dummy_df 
# dummy_df = pd.concat([df1["Test_ID"], dummy_df], axis=1)

# dummy_df[:15]

# COMMAND ----------

# # create dummy_df_odd, dummy_df_even 
# # use combine_first() 

# # Split dummy_df into two df, one with even rows, another with all the odd rows 
# dummy_df_even = dummy_df.iloc[lambda x: x.index % 2 == 0].reset_index(drop=True).replace(0, pd.NA)
# dummy_df_odd = dummy_df.iloc[lambda x: x.index % 2 != 0].reset_index(drop=True).replace(0, pd.NA)

# #.replace(0, pd.NA).

# #dummy_df_even[:15]

# combined_df = dummy_df_even.combine_first(dummy_df_odd)


# combined_df[:15]

# COMMAND ----------

# # turn the Null values back to 0's

# combined_df = combined_df.fillna(0)

# combined_df[:15]

# COMMAND ----------

# # combine df8 and combined_df 

# df9 = df8.combine_first(combined_df)

# df9[:15]

# COMMAND ----------

# # Dropping unnecessary columns 

# df9 = df9.drop(columns=["Threshold/Delay", "Threshold/Jitter"])

# df9[:15]

# COMMAND ----------

# # Reordering the columns (putting max_ratio and Quality of Experience to the end)

# df9 = df9[["Test_ID", "Delay", "Jitter", "Ratio/Delay", "Ratio/Jitter", "Audio RTP from->to 20 kbps/Delay (ms)", "Audio RTP from->to 20 kbps/Jitter (ms)", "Audio RTP to->from 20 kbps/Delay (ms)", "Audio RTP to->from 20 kbps/Jitter (ms)", "Video RTP from->to 150 kbps/Delay (ms)", "Video RTP from->to 150 kbps/Jitter (ms)", "KPI from->to/Delay (ms)", "KPI from->to/Jitter (ms)", "KPI to->from/Delay (ms)", "KPI to->from/Jitter (ms)", "Max_Ratio(Delay/Jitter)", "Quality of experience"]]
