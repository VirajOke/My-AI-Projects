# Databricks notebook source
#Setup
! pip install /Workspace/Repos/Production/dsai_databricks_helpers -q
#TensorFlow setup
! pip install --upgrade tensorflow -q
! pip install tensorflow-text -q
! pip install tf-models-official -q

# COMMAND ----------

import pandas as pd
from sklearn import preprocessing
import dsai_databricks_helpers as dsai
import statistics
import re
import os
import shutil
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from keras.callbacks import EarlyStopping
from tensorflow_addons.optimizers import AdamW
from official.nlp import optimization  # to create AdamW optimizer
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

tf.get_logger().setLevel('ERROR')

# COMMAND ----------


spark.conf.set("fs.azure.account.key.scsccsadsailabdevdls1.dfs.core.windows.net", dbutils.secrets.get(scope="storage-account-access-key", key="storage-account-access-key"))
spark.conf.set("spark.sql.parquet.mergeSchema", "true")

# COMMAND ----------

print(tf.config.list_physical_devices())
import os

if tf.config.list_physical_devices('GPU'):
    strategy = tf.distribute.MirroredStrategy()
    print('Using GPU')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the data into a DataFrame

# COMMAND ----------

bits = spark.sql('''SELECT * from bits.epet_bits''').toPandas()
bits

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data preprocessing and EDA

# COMMAND ----------

unique_val = bits['service_num'].unique()
print(sorted(unique_val))

# COMMAND ----------

Description_na_indx = list(bits[bits['Description'].isnull()].index)
Service_na_indx = list(bits[bits['Service'].isnull()].index)
cnt = 0
for index in Description_na_indx:
    if index in Service_na_indx:
        cnt += 1

print(f'Number of NAs in "Service" are: {(len(Service_na_indx))}')
print(f'Number of NAs in "Description" are: {(len(Description_na_indx))}')
print(f'Number of NA Data points in total: {(len(Description_na_indx)+(len(Service_na_indx)-cnt))}')

# COMMAND ----------

bits.dropna(inplace = True)
bits.isna().sum()

# COMMAND ----------

# data_list = []
# for data in bits['service_num']:
#     data_list.append(str(data))
# bits['service_num'] = data_list

# COMMAND ----------

target_labels = bits['service_num'].value_counts().to_dict()

# COMMAND ----------

# Renaming calsses with datapoints <=252 as '0'.
val_index = []
for keys, val_count in target_labels.items():
    if val_count<=252:
        bits.loc[bits["service_num"] == keys, ["service_num"]] = 0
        bits.loc[bits["service_num"] == 31, ["service_num"]] = 0
        bits.loc[bits["service_num"] == 0, ["Service"]] = 'Other'
        bits.drop((bits.loc[bits["service_num"] == keys]).index)

# COMMAND ----------

bits.reset_index(drop=True, inplace= True)
bits

# COMMAND ----------

le = preprocessing.LabelEncoder()
bits['service_num'] = le.fit_transform(bits['Service'])

new_target_labels = bits['service_num'].value_counts().to_dict()
x_axis = list(new_target_labels)
y_axis = list(new_target_labels.values())
df_for_stats = pd.DataFrame(y_axis, columns = ['y_axis'])

# COMMAND ----------

df_for_stats.describe()

# COMMAND ----------

new_target_labels

# COMMAND ----------

# Note:- Heavily imbalanced classes. If the model doesn't perform well after , try to balance the classes. 
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(x_axis, y_axis)
plt.show()

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Text cleaning

# COMMAND ----------

temp_text_holder = []
for text in bits['Description']:
    text = re.sub('[^A-Za-z\s+]','', text)
    text = re.sub("\s+"," ", text)
    temp_text_holder.append(text)
bits['Description'] = temp_text_holder                              

# COMMAND ----------

bits_df = bits.drop(columns=['BR_NMBR', 'Service'])
bits_df.rename(columns = {'service_num':'target'}, inplace = True)
bits_df.rename(columns = {'Description':'text'}, inplace = True)
bits_df.head()

# COMMAND ----------

for i in range(0,20):
    print(bits_df['text'][i], f'. ({i}) Text ends here -----', bits_df['target'][i])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train test split

# COMMAND ----------

# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop('target')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    print(ds)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
        ds = ds.batch(batch_size)
    return ds

# COMMAND ----------

train, test = train_test_split(bits_df, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')

# COMMAND ----------

val

# COMMAND ----------

train_ds = df_to_dataset(train)
val_ds = df_to_dataset(val)
test_ds = df_to_dataset(test)

# COMMAND ----------

for feature_batch, label_batch in train_ds.take(1):
    print('Every feature:', list(feature_batch.keys()))
    print('A batch of targets:', label_batch )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setting up TensorFlow
# MAGIC - #### Choose a model and its preprocessor

# COMMAND ----------

bert_model_name = 'bert_en_uncased_L-12_H-768_A-12'

map_name_to_handle = {
    'bert_en_uncased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3',
    'bert_en_cased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/3',
    'bert_multi_cased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/3',
    'small_bert/bert_en_uncased_L-2_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-2_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-2_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-2_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-768_A-12/1',
    'small_bert/bert_en_uncased_L-4_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-4_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-4_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-4_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-768_A-12/1',
    'small_bert/bert_en_uncased_L-6_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-6_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-6_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-6_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-768_A-12/1',
    'small_bert/bert_en_uncased_L-8_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-8_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-8_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-8_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-768_A-12/1',
    'small_bert/bert_en_uncased_L-10_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-10_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-10_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-10_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-768_A-12/1',
    'small_bert/bert_en_uncased_L-12_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-12_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-12_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-768_A-12/1',
    'albert_en_base':
        'https://tfhub.dev/tensorflow/albert_en_base/2',
    'electra_small':
        'https://tfhub.dev/google/electra_small/2',
    'electra_base':
        'https://tfhub.dev/google/electra_base/2',
    'experts_pubmed':
        'https://tfhub.dev/google/experts/bert/pubmed/2',
    'experts_wiki_books':
        'https://tfhub.dev/google/experts/bert/wiki_books/2',
    'talking-heads_base':
        'https://tfhub.dev/tensorflow/talkheads_ggelu_bert_en_base/1',
}

map_model_to_preprocess = {
    'bert_en_uncased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'bert_en_cased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_cased_preprocess/3',
    'small_bert/bert_en_uncased_L-2_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-2_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-2_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-2_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-4_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-4_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-4_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-4_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-6_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-6_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-6_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-6_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-8_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-8_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-8_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-8_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-10_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-10_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-10_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-10_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-12_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-12_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-12_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'bert_multi_cased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3',
    'albert_en_base':
        'https://tfhub.dev/tensorflow/albert_en_preprocess/3',
    'electra_small':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'electra_base':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'experts_pubmed':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'experts_wiki_books':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'talking-heads_base':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
}

tfhub_handle_encoder = map_name_to_handle[bert_model_name]
tfhub_handle_preprocess = map_model_to_preprocess[bert_model_name]

print(f'BERT model selected           : {tfhub_handle_encoder}')
print(f'Preprocess model auto-selected: {tfhub_handle_preprocess}')

# COMMAND ----------

# MAGIC %md 
# MAGIC - #### Load the BERT model and it's preprocessor

# COMMAND ----------

bert_preprocess_model = hub.KerasLayer(tfhub_handle_preprocess)
bert_model = hub.KerasLayer(tfhub_handle_encoder)

# COMMAND ----------

# MAGIC %md 
# MAGIC - #### Build classifier model

# COMMAND ----------

def build_classifier_model():
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    
    neural_net = outputs['pooled_output']
    neural_net = tf.keras.layers.Dropout(0.1)(neural_net)
    neural_net = tf.keras.layers.Dense(26, activation=None, name='classifier')(neural_net) # 32 classes 
    return tf.keras.Model(inputs = [text_input], outputs= [neural_net])

# COMMAND ----------

model = build_classifier_model()
model.summary()

# COMMAND ----------

# loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
metrics = tf.metrics.SparseCategoricalCrossentropy(name='sparse_categorical_crossentropy')

# COMMAND ----------

classifier_model = build_classifier_model()
classifier_model.compile(loss = 'SparseCategoricalCrossentropy',
                         optimizer = 'adam', 
                         metrics = ['accuracy'])

# COMMAND ----------

#es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1)

# COMMAND ----------

epochs = 90
history = classifier_model.fit(x=train_ds,
                               validation_data=val_ds,
                               epochs=epochs,
                               )

# COMMAND ----------


