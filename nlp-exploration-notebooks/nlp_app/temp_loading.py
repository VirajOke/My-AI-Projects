import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import time

from dash.dependencies import Input, Output

app = dash.Dash()

app.layout = html.Div(
    children=[
        html.H3("Click Button for loading"),
        html.Button(
            'Launch Loading',
            id='loading-button',
            className="validate-button"
        ),
        html.Div(id='loading-output'),
        dcc.Loading(type='graph',fullscreen =True, children=html.Div(
            id='loading-hidden-div', children=None, style={'display': 'none'}))
    ]
)

# Loading hidden prop included as Callback output


@app.callback([
    Output("loading-output", "children"),
    Output("loading-hidden-div", "children")],
    [Input("loading-button", "n_clicks")])
def button_triggers_loading(n_clicks):
    if n_clicks is None:
        return (None, None)
    time.sleep(5)
    return ('Loading Done', None)


if __name__ == "__main__":
    app.run_server(debug=True)