import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import urllib
import numpy as np
import dash_bootstrap_components as dbc

theme = dbc.themes.BOOTSTRAP
css = 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css'
app = dash.Dash(__name__, external_stylesheets=[theme, css])
server = app.server
# app layout with 2 columns, 1 for the parameters and 1 for the JSON export

NB_LAYERS = 1
NB_NEURONS = 10
NB_REGULARIZATION = 1

app.layout = html.Div([
    dbc.Row([
        dbc.Col([
            html.H1('Neural Network'),
            html.Hr(),
            html.H3('Parameters'),
            html.Hr(),
            html.Label('Number of Layers'),
            dbc.Input(
                id='layers',
                type='number',
                min=1,
                max=20,
                value=1,
                step=1,
            ),
            # Add as many input fields as the number of layers in the network for the number of neurons in each layer and a dropdown to select the regularization method

            html.Div(id='layer_inputs'),


            html.Label('Activation Function'),
            dcc.Dropdown(
                id='activation',
                options=[
                    
                    {'label': 'Leaky ReLU', 'value': 'LeakyReLU'},
                ],
                value='LeakyReLU'
            ),
            html.Label('Optimizer'),
            dcc.Dropdown(
                id='optimizer',
                options=[
                    {'label': 'Fixed Leaning Rate', 'value': 'FixedLearningRate'},
                    {'label': 'Momentum', 'value': 'Momentum'},
                    {'label': 'Adam', 'value': 'Adam'},
                ],
                value='Momentum'
            ),
            html.Label('Learning Rate'),
            # input field for the learning rate
            dbc.Input(
                id='learning_rate',
                type='number',
                value=0.01,
                min=0.0001,
                max=1,
                step=0.0001
            ),
            html.Label('Batch Size'),
            dbc.Input(
                id='batch_size',
                type='number',
                min=1,
                max=100,
                step=1,
                value=32
            ),

            # button to export the JSON with n_clicks=0
            html.Button('Export', id='export', n_clicks=0, style={'marginTop': 20, 'marginBottom': 20})

        ], className="col-lg-4 col-md-4 col-sm-4 col-xs-4"),
        dbc.Col([
            html.H1('Neural Network'),
            html.Hr(),
            html.H3('JSON Export'),
            html.Hr(),
            html.Div(id='json')
        ], className="col-lg-8 col-md-8 col-sm-8 col-xs-8")
    ], className="col-lg-12 col-md-12 col-sm-12 col-xs-12")
], className='container')

# callback to update the number of input fields for the number of neurons in each layer

@app.callback(
    Output('layer_inputs', 'children'),
    Input('layers', 'value')
)
def update_layer_inputs(layers):
    """
    Update the number of input fields for the number of neurons in each layer
    For each layer, add a dropdown to select the regularization method and an input field for the number of neurons
    """
    # update NB_LAYERS and NB_REGULARIZATION
    global NB_LAYERS
    global NB_REGULARIZATION
    NB_LAYERS = layers
    NB_REGULARIZATION = layers
    return [html.Div([
        html.Label('Layer {}'.format(i+1)),
        dbc.Row([
            dbc.Col([
                html.Label('Regularization'),
                dcc.Dropdown(
                    id='regularization_{}'.format(i),
                    options=[
                        {'label': 'None', 'value': 'None'},
                        {'label': 'L1 Penalty', 'value': 'L1Penalty'},
                    ],
                    value='None'
                )
            ], className="col-lg-6 col-md-6 col-sm-6 col-xs-6"),
            dbc.Col([
                html.Label('Number of neurons'),
                dbc.Input(
                    id='neurons_{}'.format(i),
                    type='number',
                    value=10,
                    min=1,
                    max=100,
                    step=1
                )
            ], className="col-lg-6 col-md-6 col-sm-6 col-xs-6")
        ], className="col-lg-12 col-md-12 col-sm-12 col-xs-12")
    ], style={'marginBottom': 20, 'marginTop': 20}, key='layer_{}'.format(i+1)) for i in range(layers)]


# callback to update the JSON export

@app.callback(
    Output('json', 'children'),
    Input('export', 'n_clicks'),
    State('layers', 'value'),
    State('activation', 'value'),
    State('optimizer', 'value'),
    State('learning_rate', 'value'),
    State('batch_size', 'value'),
    *[State('neurons_{}'.format(i), 'value') for i in range(NB_LAYERS)],
    *[State('regularization_{}'.format(i), 'value') for i in range(NB_REGULARIZATION)]
)
def update_json(n_clicks, layers, activation, optimizer, learning_rate, batch_size, *args):
    """
    Create a JSON export of the neural network parameters
    The format is as follows:
    "{
        "BatchSize": 4,
        "SerializedLayers": [
        {
            "UnderlyingSerializedLayer": 
            {
                "Bias": [ -0.92110407911744, 0.4291105297132535 ],
                "Weights": [
                [ 0.9207416869152576, -1.3018025340163655 ],
                [ 0.9380249351273636, -2.1235010089137383 ]
                ],
                "ActivatorType": "LeakyReLU",
                "GradientAdjustmentParameters": {
                "LearningRate":1,
                "Type":"FixedLearningRate"
                },
                "Type": "Standard"
            },
            "Type": "L2Penalty",
            "PenaltyCoefficient": 0.001
        },
        {
            "Bias": [ 0.9798584611488593 ],
            "Weights": [
            [ -1.1203093167219085 ],
            [ -2.3075129217210093 ]
            ],
            "ActivatorType": "LeakyReLU",
            "GradientAdjustmentParameters": {
            "LearningRate":1,
            "Type":"FixedLearningRate"
            },
            "Type": "Standard"
        }
        ]
    }"

    The weights are being initialized randomly following Xavier initialization
    """
    if n_clicks==0:
        return 'No JSON to display'
    print("Exportation of the neural network parameters")
    # create a list of dictionaries for each layer
    layers_list = []
    for i in range(layers):
        # get the number of neurons in the current layer
        neurons = args[i]
        # get the regularization method for the current layer
        regularization = args[i+layers]
        # get the weights and biases for the current layer
        weights = repr(np.random.randn(neurons, neurons-1) * np.sqrt(2/(neurons-1)))[6:-1]
        biases = repr(np.random.randn(neurons) * np.sqrt(2/(neurons-1)))[6:-1]
        # create a dictionary for the current layer
        # if the layer has L2 regularization, add the regularization method and the penalty coefficient
        if regularization == 'L2Penalty':
            layer = """{{\n\t
                \"UnderlyingSerializedLayer\": {{\n\t\t
                    \"Bias\": {},\n\t\t
                    \"Weights\": {},\n\t\t
                    \"ActivatorType\": \"{}\",\n\t\t
                    \"GradientAdjustmentParameters\": {{\n\t\t\t
                        \"LearningRate\":{},\n\t\t\t
                        \"Type\":\"{}\"\n\t\t
                    }},\n\t\t
                    \"Type\": \"{}\"\n\t
                }},\n\t
                \"Type\": \"{}\",\n\t
                \"PenaltyCoefficient\": {}\n
            }}""".format(biases, weights, activation, learning_rate, optimizer, 'Standard', regularization, 0.001)
        else:
            #{
        #     "Bias": [ 0.9798584611488593 ],
        #     "Weights": [
        #     [ -1.1203093167219085 ],
        #     [ -2.3075129217210093 ]
        #     ],
        #     "ActivatorType": "LeakyReLU",
        #     "GradientAdjustmentParameters": {
        #     "LearningRate":1,
        #     "Type":"FixedLearningRate"
        #     },
        #     "Type": "Standard"
        # }
            layer = """{{\n\t
                \"Bias\": {},\n\t
                \"Weights\": {},\n\t
                \"ActivatorType\": \"{}\",\n\t
                \"GradientAdjustmentParameters\": {{\n\t\t
                    \"LearningRate\":{},\n\t\t
                    \"Type\":\"{}\"\n\t
                }},\n\t
                \"Type\": \"{}\"\n
            }}""".format(biases, weights, activation, learning_rate, optimizer, 'Standard')
            
        # add the dictionary to the list of layers
        layers_list.append(layer)
    # create the JSON export
    json = """{{\n\t
        \"BatchSize\": {},\n\t
        \"SerializedLayers\": [\n\t\t
            {}\n\t
        ]\n
    }}""".format(batch_size, ',\n\t\t'.join(layers_list))
    return html.Div([
        html.Pre(json, style={'whiteSpace': 'pre-wrap'}),
        html.A('Download JSON', id='download-json', download="neural_network.json", href="data:text/json;charset=utf-8,"+urllib.parse.quote(json), target="_blank")
    ])
    

if __name__ == '__main__':
    server.run(debug=True)