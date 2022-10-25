"""import os 
os.system('pip install jupyter-dash')"""
import sys
from dash import Dash, html, Input, Output, ctx, State
from panel import state
import plotly.express as px
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_uploader as du
from dash.dependencies import Input, Output
import plotly.express as px
from xarray import align
from interactivefunctions import *

#sentiment_results = generate_sentiments_table()
# Build App
app = Dash(__name__,external_stylesheets=[dbc.themes.SLATE])
app.title = 'NLP Toolkit'
# Just change the upload destination folder when on a different server.
du.configure_upload(app,
                    r'C:\Users\OkeV\Documents\GitHub\nlp-exploration-notebooks\nlp_app\uploaded_files',
                    use_upload_id= True)
app.layout = html.Div([
    dbc.Card( 
        dbc.CardBody([
            dbc.Row([
                dbc.Col([drawText('NLP ToolKit', 'title')], width = 11.5),
            ],align='center', justify="center"),  
            html.Br(),
            
            dbc.Row([
                dbc.Col(dcc.Markdown('''
                            >### 1. NLP tasks
                            > Please select a task from the dropdown menu
                            ''',
                            style={
                                'color': 'white',
                                'marginLeft': '25%',
                                'marginRight': '0',
                            }
                        ), width = 5
                ),
                dbc.Col(dcc.Dropdown(['--No selection--',
                            'Sentiment Analysis',
                            'Document Summarization', 
                            'PII Detection'],
                            clearable= True,
                            searchable = True, 
                            search_value = 'Search or select from the dropdown',
                            id = 'drop-down',
                            placeholder = "Select a task"
                            ), width = 6
                        ), 
            ],align='center', justify="center"),  

            html.Br(),
            dbc.Row([
                dbc.Col(dcc.Markdown('''
                >### 2. Upload documents
                >
                > *Accepted file extensions: txt, docx, html and pdf*
                >  
                '''
                , 
                style={
                    'color': 'white',
                    'marginLeft': '25%',
                    'marginRight': '0',
                }
                ),width = 5),
                # 1) configure the upload folder
                # Self tip : Based on the task selected by the user from the dropdown menu, 
                # give new dynamic folder paths for different NLP tasks so that they dont mix up.

                # 2) Use the Upload component
                dbc.Col(html.Div(get_upload_component()), width = 6), 

            ], align = 'center',justify = "center", style={"height": "25%"}
            ),  
            html.Br(),

            dbc.Row([dbc.Col(html.Button('Analyze', id='analyze-val', 
                        n_clicks= 0, disabled = True),width = 6),
                        
                    dbc.Col(html.Button('Reset', id='reset-val', 
                        n_clicks= 0, disabled = False),width = 6)

            ], align = 'center', justify = "center"),  
            html.Br(), 

            dbc.Row([dbc.Col(width = 10, id= 'func-call')
            ],align = 'center', justify = "center"),  
            html.Br(), 

            dbc.Row([dbc.Col(dcc.Input(
                                    id='folder-path-state', 
                                    type='text',
                                    disabled= True
                            )
            )],align = 'right', justify = "right"),  
            html.Br(), 
            
        ]), color = 'dark' 
    )
]) 
# Callbacks
@app.callback(
    Output('dash-uploader', 'disabled'),
    Input('drop-down', 'value'),
    Input('reset-val', 'n_clicks')
)
def call_function(value, reset_value):
    if value == '--No selection--':
        return True
    
    elif 'reset-val' != ctx.triggered_id and value == 'Sentiment Analysis':
        return False

    elif 'reset-val' != ctx.triggered_id and value == 'Document Summarization':
        return False
    elif 'reset-val' == ctx.triggered_id:
        return True
    else:
        return True

@du.callback(
    output= [Output("analyze-val", "disabled"), 
            Output('folder-path-state', 'value')],
        id= "dash-uploader",
)
def callback_on_completion(status: du.UploadStatus, ):
    if status.is_completed == True:
        folder_path = f'C:/Users/OkeV/Documents/GitHub/nlp-exploration-notebooks/nlp_app/uploaded_files/{status.upload_id}'
        return False, folder_path

@app.callback(
    Output("func-call", "children"),
    Input("analyze-val", "disabled"),
    Input("drop-down","value"),
    Input("analyze-val","n_clicks"),
    State("folder-path-state","value"),
)
def function_trigger(button_disabled, drop_down_value, analyze_clicks, folder_path_value):
    if 'analyze-val' == ctx.triggered_id and button_disabled == False and drop_down_value == 'Sentiment Analysis':
        text = 'sentiment_analysis'
        return nlp_function_call(text, folder_path_value)

    if 'analyze-val' == ctx.triggered_id and button_disabled == False and drop_down_value == 'Document Summarization':
        text = 'document_summarization'
        return nlp_function_call(text, folder_path_value)

@app.callback(
    Output('drop-down','value'),
    Input('reset-val', 'n_clicks'))
def update(reset_value):
    if reset_value>0:
        return None
# Run app  

if __name__ == '__main__':
    app.run_server(debug=True)