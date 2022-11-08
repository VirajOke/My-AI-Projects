"""import os 
os.system('pip install jupyter-dash')"""
import sys
from dash import Dash, html, Input, Output, ctx, State, MATCH, ALL
from matplotlib.pyplot import title
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
import pyautogui
import ast
import dash
from dash.exceptions import PreventUpdate

#sentiment_results = generate_sentiments_table()
# Build App
app = Dash(__name__,external_stylesheets=[dbc.themes.SLATE])
app.title = 'NLP Toolkit'
# Just change the upload destination folder when on a different server.
du.configure_upload(app,r'uploaded_files',use_upload_id= True)
app.layout = html.Div([
    dbc.Card( 
        dbc.CardBody([
            dbc.Row([
                dbc.Col([drawText('NLP ToolKit', 'title')], id= 'page_title', width = 11.5),
            ],align='center', justify="center"),  
                
                html.Br(),
                dbc.Row([
                    dbc.Col(dcc.Markdown('''
                                >### 1. Upload documents
                                >
                                > *Accepted file extensions: txt, docx, html and pdf*
                                >  
                                '''
                            , id = 'upload-markdown'),width = 6),

                    dbc.Col(get_upload_component(), id= 'upload_func_call', width = 6), 
            ], align = 'center',justify = "center", style={"height": "25%"}),  
            html.Br(),
            dbc.Row([
                dbc.Col(dcc.Markdown('''
                                >### 2. Process your documents with AI
                                > Tasks: Topic extraction, Summarization and Sentiment analysis  
                                > *It may take a few seconds*
                                >  
                                '''
                            , id= 'process-markdown'),width = 5),
                dbc.Col(html.Button('Process', id='analyze-val', n_clicks= 0, disabled = True),width = 3),
                dbc.Col(html.Button('Reset', id='reset-val',n_clicks= 0, disabled = False),width = 3)

        ], align = 'center', justify = "center"),  
                
            html.Br(), 
            html.Br(), 
            dbc.Row([dbc.Col(
                    dcc.Loading(type='graph',fullscreen =True, children= html.Div(id= 'sum-output-area', children=None)), width = 8),
                    ],align = 'center', justify = "center"),  
            html.Br(), 
            
            html.Div(id = 'summ_modal'),
            
            dbc.Row([
                dbc.Col(dcc.Markdown('''
                                >### 3. Analyze Sentiments and PII 
                                > Sentiments: 1 Star (Most negative) / 5 Stars (Most positive)
                                > Personally Identifiable Information (PII): 
                                >  *Person, Email address, Phone number, Location*
                                '''
                            , id= 'senti-pii-markdown'),width = 5),
                dbc.Col(html.Button('Analyze Sentiments', id='analyze-sentiments-btn', n_clicks= 0, disabled = False),width = 3),
                dbc.Col(html.Button('Analyze PII', id='analyze-pii-btn', n_clicks= 0, disabled = False),width = 3)
        ], align = 'center', justify = "center", id= 'sentiments-row', style ={'visibility': 'hidden'}),  
            html.Br(),
            dbc.Row([dbc.Col(dcc.Loading(type='graph',fullscreen =True, children=html.Div(id= 'sentiment-func-call')),width = 10)]
            ,align = 'center', justify = "center"),  
            html.Br(), 
            dbc.Row([dbc.Col(width = 10, id= 'sentiment-plot-call')],align = 'center', justify = "center"),  
            html.Br(), 
            dbc.Row([dbc.Col(dcc.Loading(type='graph',fullscreen =True, children=html.Div(id= 'pii-func-call')),width = 10)]
            ,align = 'center', justify = "center"),  
            html.Br(), 
            dbc.Row([dbc.Col(dcc.Input(id='folder-path-state',type='text',disabled= True)
            )],align = 'right', justify = "right"),  
            html.Br(), 

            dbc.Row([dbc.Col(dcc.Input(id='temp_value_state',type='text',disabled= True, 
                                    style = {'display':'none'})
            )],align = 'right', justify = "right"),  
            html.Br(), 

            dbc.Row([dbc.Col(width = 2, id= 'del_files_call')],align = 'center', justify = "center"),  
            html.Br(), 

        ]), color = 'dark' 
    )
]) 
# Callbacks
# ------------ Upload componant callbacks ------------
@du.callback(
    output= [Output("analyze-val", "disabled"), 
            Output('folder-path-state', 'value')],
    id= "dash-uploader",
)
def callback_on_completion(status: du.UploadStatus):
    if status.is_completed == True:
        folder_path = f'uploaded_files/{status.upload_id}'
        return False, folder_path
        
# ------------ Function callbacks for topics and summarization ------------
@app.callback(
    Output("sum-output-area", "children"),
    Output('temp_value_state', 'value'),
    Output("sentiments-row","style"),
    Input("analyze-val","n_clicks"),
    State("folder-path-state","value"),  
)
def function_trigger(analyze_clicks, folder_path_value):
    if 'analyze-val' == ctx.triggered_id:
        title_buttons, topics_list, = nlp_tasks(folder_path_value, 'topics_func_call')
        summaries = nlp_tasks(folder_path_value, 'summarization_func_call')
        merge_comp = []
        for index, button in enumerate(title_buttons):
            merge_comp.append(button)
            merge_comp.append(html.P(str(topics_list[index]), className = 'topics-str'))
        return dbc.Card(
                dbc.CardBody([
                    dcc.Markdown('''
                                    >### Topics
                                    >
                                '''
                                ,
                                style = {'color': 'white','marginLeft': '4%', }),
                    html.Div(merge_comp)
                ]),style={'overflow-y': 'scroll','height': '50vh'}
            ), summaries, {'visibility': 'visible'}
    else:
        return None, None

@app.callback(
    Output('summ_modal', 'children'),
    Input({'type': 'dynamic-buttons', 'index': ALL}, 'n_clicks'),
    Input({'type': 'dynamic-buttons', 'index': ALL}, 'id'),
    State('temp_value_state', 'value'))
def display_output(n_clicks, id, value):
    if not any(n_clicks):
        raise PreventUpdate
    else:
        ctx = dash.callback_context
        if ctx.triggered:
            # grab the ID of the button that was triggered
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            button_id = ast.literal_eval(button_id)
            if button_id in id:
                doc_index = button_id['index'] 
                task = value[doc_index]
                return dbc.Modal([
                            dbc.ModalHeader(dbc.ModalTitle("Document Summary"), style={'color':'white'}),
                                dbc.ModalBody(html.P(task), style={'color':'white'}),
                                ], id="modal-body-scroll", centered=True, 
                                scrollable=True, is_open= True),
            else:
                return html.Div([
                    html.Div('No task executed')
                ])

# ------------ Function callback for sentiments ------------
@app.callback(
    Output("sentiment-func-call", "children"),
    Output("sentiment-plot-call", "children"),
    Output("pii-func-call", "children"),
    Input("analyze-sentiments-btn","n_clicks"),
    Input("analyze-pii-btn","n_clicks"), 
    State("folder-path-state","value"))
def function_trigger(analyze_sentiment_clicks, analyze_pii_clicks, folder_path_value):
    if 'analyze-sentiments-btn' == ctx.triggered_id:
        df, plot = sentiment_analysis(folder_path_value)
        return df, plot, None
    if 'analyze-pii-btn' == ctx.triggered_id:
        pii_df = pii_analysis(folder_path_value)
        return None, None, pii_df

# ------------ Reset callbacks ------------
@app.callback(
    Output('del_files_call', 'children'),
    Output('upload_func_call','children'),
    Input('reset-val', 'n_clicks'),
    State("folder-path-state","value")
)
def delete_files(n_clicks, session_id):
    if 'reset-val' == ctx.triggered_id:
        pyautogui.hotkey('f5')
        return delete_uploaded_files(session_id), get_upload_component()

# Run app  
if __name__ == '__main__':
    app.run_server(debug= False)

    