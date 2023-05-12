# Databricks notebook source
import json
import pandas as pd
import numpy as np
from pyspark.sql.session import SparkSession


# COMMAND ----------

def get_dbutils(spark):
    dbutils = None
    if spark.conf.get("spark.databricks.service.client.enabled") == "true":
        from pyspark.dbutils import DBUtils
        dbutils = DBUtils(spark)
    else:
        import IPython
        dbutils = IPython.get_ipython().user_ns["dbutils"]
    return dbutils

def grab_table(tablename,
               kv_name="ScScCSV-DSAI-Lab-dev-kv", 
               storage_acct="scsccsadsailabdevdls1",
               container="epet"):
    sess = SparkSession.builder.master("local").appName("Session").getOrCreate()
    dbutils = get_dbutils(sess)
    creds = dbutils.secrets.get(scope=kv_name, key="storage-account-access-key")
    sess.conf.set(f"fs.azure.account.key.{storage_acct}.dfs.core.windows.net", creds)

    try:
        dbutils.fs.mount(
            source = f"wasbs://{container}@{storage_acct}.blob.core.windows.net/",
            mount_point = f"/mnt/{container}",
            extra_configs = {f"fs.azure.account.key.{storage_acct}.blob.core.windows.net":
                                creds}
        )
    except:
        print(f'{container} probably already mounted')
        df = None
        
    #df = sess.table("smg.derived_bits").toPandas()
    #sess.stop() # this breaks Databricks :D
    return sess.table(tablename).toPandas()

epet_df = grab_table('vizhelper.epet', container='epet')
epet_df.info()

# COMMAND ----------

bits_df = grab_table('edr.demand_br_bits_items', container='edr')
bits_df.info()
# possible y = bits_df['LEAD_PROD_ID'] + bits_df['LEAD_SERVICE_ID']
X = bits_df["REQMT_OVRVW"].dropna().reset_index(drop=True)

# COMMAND ----------

# MAGIC %md
# MAGIC taken from [here](https://www.vennify.ai/bertopic-topic-modeling/)

# COMMAND ----------

# MAGIC %md
# MAGIC ```python
# MAGIC !pip install datasets
# MAGIC from datasets import load_dataset
# MAGIC dataset = load_dataset("newspop", split="train[:]")
# MAGIC 
# MAGIC docs = []
# MAGIC for case in dataset:
# MAGIC     if case['topic'] == 'economy':
# MAGIC         docs.append(case['headline'])
# MAGIC ```

# COMMAND ----------

!pip install bertopic
from bertopic import BERTopic

# default is all-MiniLM-L6-v2
topic_model = BERTopic()

topics, probs = topic_model.fit_transform(X)

# COMMAND ----------

topic_information = topic_model.get_topic_info()
print(topic_information)

# COMMAND ----------

topic_model.visualize_topics()

# COMMAND ----------

topic_model.visualize_barchart()

# COMMAND ----------

dir(topic_model)

# COMMAND ----------

# MAGIC %md
# MAGIC Scott text 
# MAGIC ```
# MAGIC I've been assuming that each cluster has an ordered list of terms, ie/ termcluster[0] is the word that best represents each cluster.
# MAGIC So I was thinking that we could take up to 3 terms from each cluster and use them as features- zero shot classification (not sure if that's the official term) 
# MAGIC could score each BR against the cluster terms. We'd end up with values we could fit into a model, ie RCMP: .92, wifi: .03. 
# MAGIC Since you folks are already on it, I'll clean my kitchen.
# MAGIC ```

# COMMAND ----------

for key, val in topic_model.get_topics().items():
    # val is a list of tuples ('keyword', confidence). Only need the keywords currently
    print([i[0] for i in val[:3]])

# COMMAND ----------

# get service lines for each BR, find topics that are associated with each.
lead_sl = bits_df["LEAD_PROD_ID"].unique()
print(f"there are a total of {lead_sl.shape[0]} service lines.")
for sl in lead_sl:
    sl_df = bits_df.loc[bits_df["LEAD_PROD_ID"]==sl, :]
    try:
        topics, probs = topic_model.transform(sl_df["REQMT_OVRVW"].dropna().reset_index(drop=True))
        print(f"For Lead Product {sl}: {sl_df.shape[0]} BRs with {np.unique(topics).shape[0]} unique topics.")
    except ValueError:
        print(f"For Lead Product {sl} something went wrong!")

# COMMAND ----------

# MAGIC %md
# MAGIC First run 2022-11-25 15:53
# MAGIC ```there are a total of 167 service lines.
# MAGIC For Lead Product 21STR_SF: 4598 BRs with 292 unique topics.
# MAGIC For Lead Product 16_MR: 3424 BRs with 335 unique topics.
# MAGIC For Lead Product 16_NCO: 1050 BRs with 214 unique topics.
# MAGIC For Lead Product 16_WAN: 990 BRs with 198 unique topics.
# MAGIC For Lead Product 21TCC_CC: 841 BRs with 75 unique topics.
# MAGIC For Lead Product None something went wrong!
# MAGIC For Lead Product 21SIS_SIS: 1208 BRs with 222 unique topics.
# MAGIC For Lead Product 16_WIFI: 505 BRs with 82 unique topics.
# MAGIC For Lead Product 21IBN_WF: 1746 BRs with 148 unique topics.
# MAGIC For Lead Product 16_CAB: 710 BRs with 117 unique topics.
# MAGIC For Lead Product 16_LAN: 407 BRs with 94 unique topics.
# MAGIC For Lead Product 21IBN_LAN: 1550 BRs with 214 unique topics.
# MAGIC For Lead Product 21FW_FW: 918 BRs with 162 unique topics.
# MAGIC For Lead Product 16_MSFT: 32 BRs with 9 unique topics.
# MAGIC For Lead Product 21IBN_CB: 1575 BRs with 172 unique topics.
# MAGIC For Lead Product 21INS_VMS: 148 BRs with 41 unique topics.
# MAGIC For Lead Product 21SMG_SMG: 80 BRs with 26 unique topics.
# MAGIC For Lead Product 21STL_MB: 75 BRs with 15 unique topics.
# MAGIC For Lead Product 21GCW_GCW: 5353 BRs with 369 unique topics.
# MAGIC For Lead Product 16_PHONE: 1009 BRs with 93 unique topics.
# MAGIC For Lead Product 21MDL_MDL: 866 BRs with 167 unique topics.
# MAGIC For Lead Product 21ICM_myK: 500 BRs with 24 unique topics.
# MAGIC For Lead Product 21TFL_VO: 800 BRs with 103 unique topics.
# MAGIC For Lead Product 21TCS_VCS: 1471 BRs with 119 unique topics.
# MAGIC For Lead Product 21WTP_SW: 1079 BRs with 90 unique topics.
# MAGIC For Lead Product 16_RPS: 609 BRs with 92 unique topics.
# MAGIC For Lead Product 21TFV_TFV: 60 BRs with 11 unique topics.
# MAGIC For Lead Product 16_MDL: 369 BRs with 121 unique topics.
# MAGIC For Lead Product 18_GCSI: 92 BRs with 24 unique topics.
# MAGIC For Lead Product 21CI_SI: 626 BRs with 51 unique topics.
# MAGIC For Lead Product 21STL_FX: 116 BRs with 30 unique topics.
# MAGIC For Lead Product 21MDR_AIX: 414 BRs with 85 unique topics.
# MAGIC For Lead Product 16_RCD: 72 BRs with 26 unique topics.
# MAGIC For Lead Product MTS: 91 BRs with 9 unique topics.
# MAGIC For Lead Product 21DCF_DCF: 229 BRs with 69 unique topics.
# MAGIC For Lead Product 16_RPL: 188 BRs with 45 unique topics.
# MAGIC For Lead Product 16_VOTH: 537 BRs with 39 unique topics.
# MAGIC For Lead Product 21MSF_MSF: 205 BRs with 37 unique topics.
# MAGIC For Lead Product MIDAHS: 1395 BRs with 52 unique topics.
# MAGIC For Lead Product 16_EICM: 63 BRs with 18 unique topics.
# MAGIC For Lead Product 16_WEBCON: 9 BRs with 3 unique topics.
# MAGIC For Lead Product OFF: 2686 BRs with 183 unique topics.
# MAGIC For Lead Product 21MFR_OL: 276 BRs with 50 unique topics.
# MAGIC For Lead Product MNS: 94 BRs with 14 unique topics.
# MAGIC For Lead Product 16_CM: 20 BRs with 9 unique topics.
# MAGIC For Lead Product 21HPC_HPC: 167 BRs with 37 unique topics.
# MAGIC For Lead Product 16_RM: 13 BRs with 7 unique topics.
# MAGIC For Lead Product 16_NMS: 44 BRs with 16 unique topics.
# MAGIC For Lead Product 16_NI: 114 BRs with 35 unique topics.
# MAGIC For Lead Product 16_TCO: 522 BRs with 81 unique topics.
# MAGIC For Lead Product 16_HPC: 63 BRs with 19 unique topics.
# MAGIC For Lead Product 21DS_DS: 75 BRs with 23 unique topics.
# MAGIC For Lead Product 16_MDR: 28 BRs with 13 unique topics.
# MAGIC For Lead Product 21MDR_LNX: 644 BRs with 111 unique topics.
# MAGIC For Lead Product VS: 547 BRs with 31 unique topics.
# MAGIC For Lead Product 21IBN_DCN: 315 BRs with 79 unique topics.
# MAGIC For Lead Product 16_DS: 31 BRs with 6 unique topics.
# MAGIC For Lead Product 21CLD_CS: 210 BRs with 39 unique topics.
# MAGIC For Lead Product 16_FS: 33 BRs with 12 unique topics.
# MAGIC For Lead Product 16_ID: 23 BRs with 8 unique topics.
# MAGIC For Lead Product RDIMS something went wrong!
# MAGIC For Lead Product 16_ICM: 105 BRs with 15 unique topics.
# MAGIC For Lead Product 16_IP: 131 BRs with 45 unique topics.
# MAGIC For Lead Product 16_MSS: 22 BRs with 11 unique topics.
# MAGIC For Lead Product 16_PSE: 37 BRs with 18 unique topics.
# MAGIC For Lead Product 16_EMS: 10 BRs with 5 unique topics.
# MAGIC For Lead Product 21TMD_EMD: 144 BRs with 18 unique topics.
# MAGIC For Lead Product 21TMD_MD: 18 BRs with 9 unique topics.
# MAGIC For Lead Product 16_DB: 43 BRs with 20 unique topics.
# MAGIC For Lead Product 21NWM_NWM: 116 BRs with 44 unique topics.
# MAGIC For Lead Product 16_VMS: 96 BRs with 17 unique topics.
# MAGIC For Lead Product 16_SMS: 45 BRs with 14 unique topics.
# MAGIC For Lead Product 21DB_DB: 134 BRs with 40 unique topics.
# MAGIC For Lead Product 16_DCFM: 25 BRs with 10 unique topics.
# MAGIC For Lead Product 16_NP: 30 BRs with 12 unique topics.
# MAGIC For Lead Product MWSS: 332 BRs with 8 unique topics.
# MAGIC For Lead Product WHS: 78 BRs with 4 unique topics.
# MAGIC For Lead Product COL: 62 BRs with 5 unique topics.
# MAGIC For Lead Product NGSRA: 17 BRs with 2 unique topics.
# MAGIC For Lead Product 16_NONE: 65 BRs with 29 unique topics.
# MAGIC For Lead Product MFS: 8 BRs with 3 unique topics.
# MAGIC For Lead Product MFIRE: 46 BRs with 1 unique topics.
# MAGIC For Lead Product MSFT: 169 BRs with 7 unique topics.
# MAGIC For Lead Product ACCKEY something went wrong!
# MAGIC For Lead Product ARMS something went wrong!
# MAGIC For Lead Product EMS: 33 BRs with 1 unique topics.
# MAGIC For Lead Product DNS something went wrong!
# MAGIC For Lead Product SCNET: 84 BRs with 3 unique topics.
# MAGIC For Lead Product 21TFL_PBX: 51 BRs with 12 unique topics.
# MAGIC For Lead Product GMCS: 619 BRs with 35 unique topics.
# MAGIC For Lead Product 16_TELCON: 71 BRs with 13 unique topics.
# MAGIC For Lead Product MSS something went wrong!
# MAGIC For Lead Product MASS something went wrong!
# MAGIC For Lead Product SMANS: 123 BRs with 12 unique topics.
# MAGIC For Lead Product MURLF something went wrong!
# MAGIC For Lead Product ICMS: 279 BRs with 5 unique topics.
# MAGIC For Lead Product CMS something went wrong!
# MAGIC For Lead Product FSS: 61 BRs with 4 unique topics.
# MAGIC For Lead Product MURL: 5 BRs with 1 unique topics.
# MAGIC For Lead Product VPN: 28 BRs with 2 unique topics.
# MAGIC For Lead Product GCDOCS something went wrong!
# MAGIC For Lead Product MAINAHS: 141 BRs with 6 unique topics.
# MAGIC For Lead Product CIT something went wrong!
# MAGIC For Lead Product GEDS something went wrong!
# MAGIC For Lead Product ITSC: 23 BRs with 2 unique topics.
# MAGIC For Lead Product UPSS something went wrong!
# MAGIC For Lead Product CNS2: 97 BRs with 12 unique topics.
# MAGIC For Lead Product CS: 29 BRs with 3 unique topics.
# MAGIC For Lead Product IDS: 14 BRs with 1 unique topics.
# MAGIC For Lead Product GNES: 284 BRs with 17 unique topics.
# MAGIC For Lead Product GTS: 15 BRs with 3 unique topics.
# MAGIC For Lead Product LAS: 27 BRs with 6 unique topics.
# MAGIC For Lead Product MCSG something went wrong!
# MAGIC For Lead Product SN10000: 9 BRs with 2 unique topics.
# MAGIC For Lead Product NSS something went wrong!
# MAGIC For Lead Product GEDIS something went wrong!
# MAGIC For Lead Product SEC something went wrong!
# MAGIC For Lead Product HISRCH something went wrong!
# MAGIC For Lead Product TFVS: 5 BRs with 2 unique topics.
# MAGIC For Lead Product WCS: 85 BRs with 13 unique topics.
# MAGIC For Lead Product MAVS something went wrong!
# MAGIC For Lead Product CAR: 59 BRs with 2 unique topics.
# MAGIC For Lead Product CNS something went wrong!
# MAGIC For Lead Product ECMS something went wrong!
# MAGIC For Lead Product CCS: 4 BRs with 3 unique topics.
# MAGIC For Lead Product 16_SOL: 11 BRs with 4 unique topics.
# MAGIC For Lead Product ENS: 1 BRs with 1 unique topics.
# MAGIC For Lead Product 16_SD: 11 BRs with 4 unique topics.
# MAGIC For Lead Product 16_CIA: 4 BRs with 2 unique topics.
# MAGIC For Lead Product 16_SG: 19 BRs with 13 unique topics.
# MAGIC For Lead Product 16_PS: 11 BRs with 5 unique topics.
# MAGIC For Lead Product 16_GCI: 1 BRs with 1 unique topics.
# MAGIC For Lead Product 16_PAC: 4 BRs with 3 unique topics.
# MAGIC For Lead Product 16_SE: 11 BRs with 5 unique topics.
# MAGIC For Lead Product 16_ES: 6 BRs with 4 unique topics.
# MAGIC For Lead Product 21MDR_X86: 1677 BRs with 188 unique topics.
# MAGIC For Lead Product 16_SP: 14 BRs with 7 unique topics.
# MAGIC For Lead Product 16_SO: 3 BRs with 3 unique topics.
# MAGIC For Lead Product 21TFL_CX: 128 BRs with 36 unique topics.
# MAGIC For Lead Product 16_TAC: 1 BRs with 1 unique topics.
# MAGIC For Lead Product 21NC_NC: 109 BRs with 31 unique topics.
# MAGIC For Lead Product 21NC_PM: 115 BRs with 32 unique topics.
# MAGIC For Lead Product 21ADV_ADV: 47 BRs with 14 unique topics.
# MAGIC For Lead Product 21INS_ISE: 49 BRs with 17 unique topics.
# MAGIC For Lead Product 21TMD_IBC: 77 BRs with 12 unique topics.
# MAGIC For Lead Product 21TFL_FL: 104 BRs with 22 unique topics.
# MAGIC For Lead Product 21MFR_MCP: 27 BRs with 11 unique topics.
# MAGIC For Lead Product 21IC_IC: 30 BRs with 10 unique topics.
# MAGIC For Lead Product 21SRA_SRA: 47 BRs with 11 unique topics.
# MAGIC For Lead Product 21FAS_FAS: 6 BRs with 5 unique topics.
# MAGIC For Lead Product 21EMAIL: 19 BRs with 10 unique topics.
# MAGIC For Lead Product 21INS_SIE: 7 BRs with 4 unique topics.
# MAGIC For Lead Product 21INT_INT: 21 BRs with 7 unique topics.
# MAGIC For Lead Product 21BP_BP: 5 BRs with 3 unique topics.
# MAGIC For Lead Product 21WTS_WTS: 298 BRs with 40 unique topics.
# MAGIC For Lead Product 16_GCSRA: 1 BRs with 1 unique topics.
# MAGIC For Lead Product 21MSS_MSS: 5 BRs with 3 unique topics.
# MAGIC For Lead Product 21CAS_CAS: 1 BRs with 1 unique topics.
# MAGIC For Lead Product 21NC_PMR: 89 BRs with 25 unique topics.
# MAGIC For Lead Product 21INS_EVS: 35 BRs with 10 unique topics.
# MAGIC For Lead Product 21MDR_SV: 2 BRs with 1 unique topics.
# MAGIC For Lead Product 21SMG_ISA: 1 BRs with 1 unique topics.
# MAGIC For Lead Product 21SIS_SA something went wrong!
# MAGIC For Lead Product 21ECM_GCK: 9 BRs with 4 unique topics.
# MAGIC For Lead Product 21WTP_HW: 4 BRs with 3 unique topics.
# MAGIC For Lead Product 21SRA_SRV: 8 BRs with 5 unique topics.
# MAGIC For Lead Product 21PAM_PAM: 5 BRs with 2 unique topics.
# MAGIC ```

# COMMAND ----------


