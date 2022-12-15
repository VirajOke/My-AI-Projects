from dash import Dash, dcc, html, Input, Output, State, MATCH, ALL, ctx
import ast
import dash
from dash.exceptions import PreventUpdate

app = Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div([
    html.Button("Add Filter", id="add-filter", n_clicks=0),
    html.Div(id='dropdown-container', children=[]),
    html.Div(id='dropdown-container-output'),
    html.Div(dcc.Input(id='temp_value_state',type='text',disabled= True, style = {'display':'none'}))
])

@app.callback(
    Output('dropdown-container', 'children'),
    Output('temp_value_state', 'value'),
    Input('add-filter', 'n_clicks'),
    State('dropdown-container', 'children'))
def display_dropdowns(n_clicks, children):
    name_list = ['sentiment', 'summ', 'pii']
    for indx, values in enumerate(name_list):
        new_button = html.Button(values,
            id={
                'type': 'dynamic-buttons',
                'index': indx
            }
        )
        children.append(new_button)
    return children, name_list

@app.callback(
    Output('dropdown-container-output', 'children'),
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
                return html.Div([
                    html.Div('Executing task {}'.format(task))
                ])
            else:
                return html.Div([
                    html.Div('No task executed')
                ])

if __name__ == '__main__':
    app.run_server(debug=True)
