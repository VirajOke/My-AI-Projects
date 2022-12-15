from turtle import color
from dash import Dash, html, dash_table
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from nlptoolkit import text_from_dir
import dash_bootstrap_components as dbc
import plotly.express as px
import dash_uploader as du
import dash_core_components as dcc
import uuid

def nlp_function_call(text, folder_path):
    from nlptoolkit import get_sentiment

    if text == 'sentiment_analysis':
        final_data = text_from_dir(folder_path)
        sentiments_df, plots = get_sentiment(final_data)
        le= LabelEncoder()
        sentiments_df['sentiments']= le.fit_transform(sentiments_df['sentiments'])

        fig = px.pie(sentiments_df, 
                    values = 'sentiments',
                    names = 'sentiments',
                    title= 'Document sentiments',
                    ).update_layout(
                        template='plotly_dark',
                        plot_bgcolor= 'rgba(0, 0, 0, 0)',
                        paper_bgcolor= 'rgba(0, 0, 0, 0)',
                    )

        return html.Div([
            dbc.Card(
                dbc.CardBody([
                    dash_table.DataTable(sentiments_df.to_dict('records'), 
                    [{"name": i, "id": i} for i in sentiments_df.columns],
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
        ]), html.Div([
                dbc.Card(
                dbc.CardBody([
                    dcc.Graph(figure = fig, 
                            config={
                                'displayModeBar': False
                            }) 
                        ])
                    ),  
            ])


    elif text== 'document_summarization':
        formated_text = ''
        from nlptoolkit import get_summary
        final_data = text_from_dir(folder_path)
        summaries = get_summary(final_data)

        return dbc.Card(
                dbc.CardBody([
                    dcc.Markdown('''
                        >### Document Summaries
                        >*key:value = Document name: Summary*
                        >
                    '''
                    ),
                    html.P(str(summaries))
                ])
            )
          

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
                disabled= True,
                filetypes= ['docx', 'pdf', 'html', 'txt'],
                max_file_size=1024,
                default_style= {
                        'width': '60%',
                        'height': 'auto',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'marginLeft': '30%',
                        'marginRight': '10%',
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



"""html.Div([
                html.Label("This is a sample", 
                    style={
                        'marginLeft': 'auto',
                        'marginRight': 'auto',
                }),
                html.Br(),
                
                dash_table.DataTable(sentiments_df.to_dict('records'), 
                [{"name": i, "id": i} for i in sentiments_df.columns],
                    style_cell = {
                        'overflow': 'hidden',
                        'textOverflow': 'ellipsis',
                        'maxWidth': 0
                    },
                    style_table={
                        'height': '300px', 'overflowY': 'auto',
                        'marginLeft': 'auto',
                        'marginRight': 'auto',
                        'maxWidth': '70%'
                    }
                )
            ])
"""
