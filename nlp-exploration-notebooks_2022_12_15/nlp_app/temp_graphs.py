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
import pyautogui
import ast
import dash
from dash.exceptions import PreventUpdate

final_data = text_from_dir('C:/Users/OkeV/Documents/GitHub/nlp-exploration-notebooks/text_summarization') 
sentiments_df, plots = get_sentiment(final_data)
sentiments_df.loc[sentiments_df["sentiments"] == "1 star", "sentiments"] = 1
sentiments_df.loc[sentiments_df["sentiments"] == "2 stars", "sentiments"] = 2
sentiments_df.loc[sentiments_df["sentiments"] == "3 stars", "sentiments"] = 3
sentiments_df.loc[sentiments_df["sentiments"] == "4 stars", "sentiments"] = 4
sentiments_df.loc[sentiments_df["sentiments"] == "5 stars", "sentiments"] = 5
print(sentiments_df)

sentiments_df = sentiments_df.groupby(['document_name','sentiments'], as_index= False).sentiments.value_counts()
print(sentiments_df)
sentiments_df['sentiments'] = sentiments_df['sentiments'].astype(str)
app = Dash(__name__,external_stylesheets=[dbc.themes.SLATE])
app.title = 'NLP Toolkit'
app.layout = dcc.Graph(id='bar-chart',
                      figure=px.bar(
                          data_frame= sentiments_df,
                          x= 'document_name',
                          y= 'count',
                          color = 'sentiments',
                          barmode="group",
                      ),
                    style = {
                        "width": '40%',
                        "overflow-x": "scroll"
                    }
)
# Run app  
if __name__ == '__main__':
    app.run_server(debug= True)

    