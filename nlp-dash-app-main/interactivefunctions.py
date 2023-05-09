from turtle import color
from dash import Dash, html, dash_table, dcc
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import dash_bootstrap_components as dbc
import plotly.express as px
import dash_uploader as du
import uuid
import re
from nlptoolkit import text_from_dir
from nlptoolkit import get_summary
from nlptoolkit import get_sentiment
from nlptoolkit import get_pii
import nltk
import os 
nltk.download('punkt')
nltk.download('omw-1.4')
nltk.download('stopwords')  
nltk.download('wordnet')
#os.system('python -m spacy download en_core_web_lg')

def nlp_tasks(folder_path, task):
    if task == 'topics_func_call':
        from nlptoolkit import get_topics
        data_cleaning = True
        final_data = text_from_dir(folder_path, data_cleaning)
        final_topics = get_topics(final_data)
        #print(final_topics)

        topics_list = []
        titile_list = []
        title_buttons = []
        for indx, values in enumerate(final_topics.items()):
            topics_list.append(values[1])
            regex = r'[^A-Za-z0-9]'
            title_text = str(values[0])
            title_text = re.sub(regex, " ", title_text)
            title_text = title_text.replace(" ", "_")
            titile_list.append(title_text)
            buttons = html.Button(title_text, className= 'link-btn',
                id={
                    'type': 'dynamic-buttons',
                    'index': indx
                }
            )
            title_buttons.append(buttons)
        return title_buttons, topics_list

    elif task == 'summarization_func_call':
        final_data = text_from_dir(folder_path)
        summaries = get_summary(final_data)
        summ_list = list(summaries.values())
        return summ_list

# Text field
def drawText(text, text_type):
    if text_type == 'title':
        return html.Div([
                    dbc.Card(
                        dbc.CardBody([
                             html.Div([html.H2(text),
                                ], style={'textAlign': 'center', 'color': 'white'}) 
                            ])
                        ),
                    ])
    elif text_type == '':
        return    

def get_upload_component():
    return du.Upload(
                id= 'dash-uploader',
                text= 'Drag and Drop or Select Files',
                text_completed = 'Uploaded: ',
                text_disabled='The uploader is disabled.',
                cancel_button=True,
                pause_button=False,
                disabled= False,
                filetypes= ['docx', 'pdf', 'html', 'txt'],
                max_file_size=1024,
                default_style= {
                        'width': '80%', 
                        'height': 'auto',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'marginLeft': '0%',
                        'marginRight': '20%',
                        'marginTop': '3%',
                        'color': 'white'
                    },
                max_files = 10,
                upload_id = uuid.uuid1()
            )

def delete_uploaded_files(path):
    import os
    import shutil
    if os.path.exists(path):
        try:
            shutil.rmtree(path)
        except OSError as e:
            print(f"Error:{e}")
    else:    ## Show an error ##
        print("Uploaded folder is empty")

def sentiment_analysis(folder_path):
    final_data = text_from_dir(folder_path)
    sentiments_df, plots = get_sentiment(final_data)
    #le= LabelEncoder()
    #sentiments_df['sentiments']= le.fit_transform(sentiments_df['sentiments'])
    sentiments_df.loc[sentiments_df["sentiments"] == "1 star", "sentiments" ] = 1
    sentiments_df.loc[sentiments_df["sentiments"] == "2 stars", "sentiments"] = 2
    sentiments_df.loc[sentiments_df["sentiments"] == "3 stars", "sentiments"] = 3
    sentiments_df.loc[sentiments_df["sentiments"] == "4 stars", "sentiments"] = 4
    sentiments_df.loc[sentiments_df["sentiments"] == "5 stars", "sentiments"] = 5

    return html.Div([
            dbc.Card(
                dash_table.DataTable(
                    id='datatable-interactivity',
                    columns=[
                        {"name": i, "id": i, "deletable": True, "selectable": True, "hideable": True}
                        if i == "iso_alpha3" or i == "year" or i == "id"
                        else {"name": i, "id": i, "deletable": True, "selectable": True}
                        for i in sentiments_df.columns
                    ],
                    data= sentiments_df.to_dict('records'),  # the contents of the table
                    editable=True,              # allow editing of data inside all cells
                    filter_action="native",     # allow filtering of data by user ('native') or not ('none')
                    sort_action="native",       # enables data to be sorted per-column by user or not ('none')
                    sort_mode="single",         # sort across 'multi' or 'single' columns
                    column_selectable="multi",  # allow users to select 'multi' or 'single' columns
                    row_selectable="multi",     # allow users to select 'multi' or 'single' rows
                    row_deletable=True,         # choose if user can delete a row (True) or not (False)
                    selected_columns=[],        # ids of columns that user selects
                    selected_rows=[],           # indices of rows that user selects
                    page_action="native",       # all data is passed to the table up-front or not ('none')
                    page_current=0,             # page number that user is on
                    page_size=6,                # number of rows visible per page
                    style_cell = { 
                            'overflow': 'hidden',
                            'textOverflow': 'ellipsis',
                            'maxWidth': 0
                    },
                    style_table = {
                            'height': '300px', 'overflowY': 'auto',
                            'marginLeft': 'auto',
                            'marginRight': 'auto',
                            'maxWidth': '100%'
                    },
                    style_header = {
                            'color': 'black',
                            'fontWeight': 'bold'
                    },
                    style_data = {
                            'backgroundColor': 'rgb(234, 236, 238)',
                            'color': 'black'
                    },
                    
                ))
            ])

def pii_analysis(folder_path):
    final_data = text_from_dir(folder_path)
    pii_df = get_pii(final_data)
    #le= LabelEncoder()
    #sentiments_df['sentiments']= le.fit_transform(sentiments_df['sentiments'])

    return html.Div([
            dbc.Card(
                dbc.CardBody([
                    dash_table.DataTable(pii_df.to_dict('records'), 
                    [{"name": i, "id": i} for i in pii_df.columns],
                        style_cell = {
                            'overflow': 'hidden',
                            'textOverflow': 'ellipsis',
                            'maxWidth': 0
                        },
                        style_table = {
                            'height': '300px', 'overflowY': 'auto',
                            'marginLeft': 'auto',
                            'marginRight': 'auto',
                            'maxWidth': '100%'
                        },
                        style_header = {
                            'color': 'black',
                            'fontWeight': 'bold'
                        },
                        style_data = {
                            'backgroundColor': 'rgb(234, 236, 238)',
                            'color': 'black'
                        },
                     ) 
                ])
            ),      
        ])