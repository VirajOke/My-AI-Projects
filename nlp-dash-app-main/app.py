"""import os 
os.system('pip install jupyter-dash')"""
import sys
from dash import Dash, html, Input, Output, ctx, State, MATCH, ALL, dcc
from matplotlib.pyplot import title
#from panel import state
import plotly.express as px
import dash_bootstrap_components as dbc
import dash_uploader as du
from dash.dependencies import Input, Output
import plotly.express as px
#from xarray import align
from interactivefunctions import *
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
                dbc.Col(html.A(html.Button('Reset', id='reset-val',n_clicks= 0, disabled = False), href= '/'),width = 3)

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
            dbc.Row([dbc.Col(dcc.Loading(type='graph',fullscreen =True, children=html.Div(id= 'sentiment-pii-func-call')),width = 10)]
            ,align = 'center', justify = "center"),  
            html.Br(), 
            dbc.Row([dbc.Col(width = 10, id= 'sentiment-plot-call')],align = 'center', justify = "center"),  
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

# ------------ Function callback for sentiments an pii dfs------------
@app.callback(
    Output("sentiment-pii-func-call", "children"),
    Input("analyze-sentiments-btn","n_clicks"),
    Input("analyze-pii-btn","n_clicks"), 
    State("folder-path-state","value"))
def function_trigger(analyze_sentiment_clicks, analyze_pii_clicks, folder_path_value):
    if 'analyze-sentiments-btn' == ctx.triggered_id:
        sentiment_df = sentiment_analysis(folder_path_value)
        return sentiment_df
    if 'analyze-pii-btn' == ctx.triggered_id:
        pii_df = pii_analysis(folder_path_value)
        return pii_df 
#------------- Interactive table?bar grapgh -----------------
@app.callback(
    Output(component_id='sentiment-plot-call', component_property='children'),
    [Input(component_id='datatable-interactivity', component_property="derived_virtual_data"),
     Input(component_id='datatable-interactivity', component_property='derived_virtual_selected_rows'),
     Input(component_id='datatable-interactivity', component_property='derived_virtual_selected_row_ids'),
     Input(component_id='datatable-interactivity', component_property='selected_rows'),
     Input(component_id='datatable-interactivity', component_property='derived_virtual_indices'),
     Input(component_id='datatable-interactivity', component_property='derived_virtual_row_ids'),
     Input(component_id='datatable-interactivity', component_property='active_cell'),
     Input(component_id='datatable-interactivity', component_property='selected_cells')]
)
def update_bar(all_rows_data, slctd_row_indices, slct_rows_names, slctd_rows,
               order_of_rows_indices, order_of_rows_names, actv_cell, slctd_cell):
    print('***************************************************************************')
    print('Data across all pages pre or post filtering: {}'.format(all_rows_data))
    print('---------------------------------------------')
    print("Indices of selected rows if part of table after filtering:{}".format(slctd_row_indices))
    print("Names of selected rows if part of table after filtering: {}".format(slct_rows_names))
    print("Indices of selected rows regardless of filtering results: {}".format(slctd_rows))
    print('---------------------------------------------')
    print("Indices of all rows pre or post filtering: {}".format(order_of_rows_indices))
    print("Names of all rows pre or post filtering: {}".format(order_of_rows_names))
    print("---------------------------------------------")
    print("Complete data of active cell: {}".format(actv_cell))
    print("Complete data of all selected cells: {}".format(slctd_cell))

    dff = pd.DataFrame(all_rows_data)
    dff = dff.groupby(['document_name','sentiments'], as_index= False).sentiments.value_counts()
    dff['sentiments'] = dff['sentiments'].astype(str)
    print(dff)
    
    # used to highlight selected countries on bar chart
    colors = ['#7FDBFF' if i in slctd_row_indices else '#0074D9'
              for i in range(len(dff))]

    if "sentiments" in dff and "document_name" in dff:
        return [
            dcc.Graph(id='bar-chart',
                      figure=px.bar(
                          data_frame= dff,
                          x= 'document_name',
                          y= 'count',
                          color = 'sentiments',
                          barmode="group",
                      ),
                    style = {
                        "overflow-x": "scroll"
                    }
            )
        ]
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
#dev
"""if __name__ == '__main__':
    app.run_server(debug= False)"""
    
#prod
"""try:
    if sys.argv[1] == '--serve':
        app.run_server(debug=False, host='0.0.0.0', port='80')
    else:
        app.run_server(debug=True)
except:
    app.run_server(debug=True)"""

if sys.argv[1] == '--serve':
    app.run_server(debug=False, host='0.0.0.0', port='80')
 
