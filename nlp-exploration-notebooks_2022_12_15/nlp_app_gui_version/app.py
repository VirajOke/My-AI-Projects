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
du.configure_upload(app,r'uploaded_files', use_upload_id= True)
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
                dbc.Col(dcc.Dropdown([
                                '--No selection--',
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
                            ,style={
                                'color': 'white',
                                'marginLeft': '25%',
                                'marginRight': '0',
                            }
                        ),width = 5),

                # 1) configure the upload folder
                # Self tip : Based on the task selected by the user from the dropdown menu, 
                # give new dynamic folder paths for different NLP tasks so that they dont mix up.

                # 2) Use the Upload component
                dbc.Col(get_upload_component(), id= 'upload_func_call', width = 6), 

            ], align = 'center',justify = "center", style={"height": "25%"}),  
            html.Br(),

            dbc.Row([
                dbc.Col(html.Button('Analyze', id='analyze-val', n_clicks= 0, disabled = True
                        ),width = 6),
                        
                dbc.Col(html.Button('Reset', id='reset-val', 
                        n_clicks= 0, disabled = False),width = 6)

            ], align = 'center', justify = "center"),  
            html.Br(), 

            dbc.Row([dbc.Col(width = 10, id= 'func-call')
            ],align = 'center', justify = "center"),  
            html.Br(), 

            dbc.Row([dbc.Col(width = 10, id= 'plot-call')
            ],align = 'center', justify = "center"),  
            html.Br(), 

            dbc.Row([dbc.Col(dcc.Input(id='folder-path-state', 
                                type='text',
                                disabled= True
                    )
            )],align = 'right', justify = "right"),  
            html.Br(), 

            dbc.Row([dbc.Col(width = 2, id= 'del_files_call')
            ],align = 'center', justify = "center"),  
            html.Br(), 
            
        ]), color = 'dark' 
    )
]) 
# Callbacks
# ------------ Upload componant callbacks ------------
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
        folder_path = f'uploaded_files/{status.upload_id}'
        return False, folder_path

# ------------ Function callbacks ------------
@app.callback(
    Output("func-call", "children"),
    Output("plot-call", "children"),
    Input("analyze-val", "disabled"),
    Input("drop-down","value"),
    Input("analyze-val","n_clicks"),
    State("folder-path-state","value"),
)
def function_trigger(button_disabled, drop_down_value, analyze_clicks, folder_path_value):
    if analyze_clicks>0 and button_disabled == False and drop_down_value == 'Sentiment Analysis':
        text = 'sentiment_analysis'
        df, plot = nlp_function_call(text, folder_path_value)
        return df, plot

    elif analyze_clicks>0 and button_disabled == False and drop_down_value == 'Document Summarization':
        text = 'document_summarization'
        df = nlp_function_call(text, folder_path_value)
        return df, None

# ------------ Reset callbacks ------------
@app.callback(
    Output('drop-down','value'),
    Input('reset-val', 'n_clicks'))
def update(reset_value):
    if 'reset-val' == ctx.triggered_id:
        return None

@app.callback(
    Output('del_files_call', 'children'),
    Output('upload_func_call','children'),
    Input('reset-val', 'n_clicks'),
    State("folder-path-state","value")
)
def delete_files(reset_value, session_id):
    if 'reset-val' == ctx.triggered_id:
        return delete_uploaded_files(session_id), get_upload_component()

# Run app  
if __name__ == '__main__':
    app.run_server(debug= False)

    