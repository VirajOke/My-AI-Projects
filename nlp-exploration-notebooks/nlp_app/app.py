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
import time
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
                dbc.Col([drawText('NLP ToolKit', 'title')], width = 11.5),
            ],align='center', justify="center"),  
                
                html.Br(),
                dbc.Row([
                    dbc.Col(dcc.Markdown('''
                                >### 1. Upload documents
                                >
                                > *Accepted file extensions: txt, docx, html and pdf*
                                >  
                                '''
                            , id= 'upload-markdown'),width = 5),

                    dbc.Col(get_upload_component(), id= 'upload_func_call', width = 6), 
            ], align = 'center',justify = "center", style={"height": "25%"}),  
            html.Br(),
            html.Br(),
            dbc.Row([
                dbc.Col(html.Button('Analyze', id='analyze-val', n_clicks= 0, disabled = True),width = 6),
                dbc.Col(html.Button('Reset', id='reset-val',n_clicks= 0, disabled = False),width = 6)

            ], align = 'center', justify = "center"),  
                
            html.Br(), 

            dbc.Row([dbc.Col(width = 8, id= 'sum-btn-area')
            ],align = 'center', justify = "center"),  
            html.Br(), 
            
            dbc.Row([dbc.Col(width = 8, id= 'sum-output-area')
            ],align = 'center', justify = "center"),  
            html.Br(), 

            html.Div(id = 'summ_modal'),

            dbc.Row([dbc.Col(width = 10, id= 'func-call')],align = 'center', justify = "center"),  
            html.Br(), 
            dbc.Row([dbc.Col(width = 10, id= 'plot-call')],align = 'center', justify = "center"),  
            html.Br(), 
            dbc.Row([dbc.Col(dcc.Input(id='folder-path-state',type='text',disabled= True)
            )],align = 'right', justify = "right"),  
            html.Br(), 

            dbc.Row([dbc.Col(dcc.Input(id='temp_value_state',type='text',disabled= True, 
                                    style = {'display':'none'})
            )],align = 'right', justify = "right"),  
            html.Br(), 
            
            html.Div(id='loading-output'),
            dcc.Loading(type='graph',fullscreen =True, children=html.Div(
            id='loading-hidden-div', children=None, style={'display': 'none'})),

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

# ------------ Function callback for topics ------------
"""@app.callback(
    Output("func-call", "children"),
    Input("analyze-val","n_clicks"),
    State("folder-path-state","value"),
)
def function_trigger(analyze_clicks, folder_path_value):
    if 'analyze-val' == ctx.triggered_id:
        topics = nlp_tasks(folder_path_value)
        return topics"""
# ------------ Function callbacks for summarization ------------
@app.callback(
    Output("sum-btn-area", "children"),
    Output('temp_value_state', 'value'),
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
            ), summaries

            
@app.callback(
    Output("loading-hidden-div", "children"),
    [Input("analyze-val","n_clicks")])
def button_triggers_loading(n_clicks):
    if 'analyze-val' == ctx.triggered_id:
        time.sleep(5)
    else: 
        return (None)

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


# ------------ Reset callbacks ------------
@app.callback(
    Output('del_files_call', 'children'),
    Output('upload_func_call','children'),
    Input('reset-val', 'n_clicks'),
    State("folder-path-state","value")
)
def delete_files(n_clicks, session_id):
    if 'reset-val' == ctx.triggered_id:
        return delete_uploaded_files(session_id), get_upload_component()


# Run app  
if __name__ == '__main__':
    app.run_server(debug= False)

    