import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash_table import DataTable, FormatTemplate

import pandas as pd

import pathlib
import glob
import os

import models
import base64
from PIL import Image
import io

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.title = "Explainable AI Dashboard"

server = app.server
app.config.suppress_callback_exceptions = True

# Path
BASE_PATH = pathlib.Path(__file__).parent.resolve()
DATA_PATH = BASE_PATH.joinpath("data").resolve()

# Read data
# df = pd.read_csv(DATA_PATH.joinpath("clinical_analytics.csv"))
default_img = app.get_asset_url('IMG_0459.jpg')
#print(default_img)

slider_count = 3
d_test = {'col1': ["Katze", "Hund"], 'col2': [22.9, 19.02]}
df_test = pd.DataFrame(d_test)

dataset_list = [
    'Imagenet',
    'Cats & Dogs',
]

model_labels = [
    {'label': 'Xception', 'value': 'xception'},
    {'label': 'ResNet50', 'value': 'resnet50'},
    {'label': 'MobileNetV2', 'value': 'mobilenetv2'},
    {'label': 'VGG16', 'value': 'vgg16'},
]

heatmap_labels = [
    {'label': 'jet', 'value': 'jet'},
    {'label': 'flag', 'value': 'flag'},
    {'label': 'prism', 'value': 'prism'},
    {'label': 'ocean', 'value': 'ocean'},
    {'label': 'gist earth', 'value': 'gist_earth'},
    {'label': 'terrain', 'value': 'terrain'},
    {'label': 'gist_stern', 'value': 'gist_stern'},
    {'label': 'gnuplot', 'value': 'gnuplot'},
    {'label': 'gnuplot2', 'value': 'gnuplot2'},
    {'label': 'CMRmap', 'value': 'CMRmap'},
    {'label': 'cubehelix', 'value': 'cubehelix'},
    {'label': 'brg', 'value': 'brg'},
    {'label': 'gist rainbow', 'value': 'gist_rainbow'},
    {'label': 'rainbow', 'value': 'rainbow'},
    {'label': 'turbo', 'value': 'turbo'},
    {'label': 'nipy spectral', 'value': 'nipy_spectral'},
    {'label': 'gist ncar', 'value': 'gist_ncar'},
]

gradcam_labels = [
    {'label': 'Grad Cam', 'value': 'grad_cam'},
    {'label': 'Counterfactual explanation (heatmap)', 'value': 'c_expl_heatmap'},
]

image_directory = 'Dashboard/assets/'
list_of_images = [os.path.basename(x) for x in glob.glob('{}*.jpg'.format(image_directory))]

list_2 = list_of_images


# markdown pop-up
def make_modal():
    
    try:
        with open("assets/explanation.md", "r") as f:
            readme_md = f.read()
    except:
        try:
            with open("Dashboard/assets/explanation.md", "r") as f:  # darwin/mac
                readme_md = f.read()
        except:
            with open("assets\explanation.md", "r") as f:
                readme_md = f.read()

    return html.Div(
        id="markdown",
        className="modal",
        style={"display": "none"},
        children=[
            html.Div(
                id="markdown-container",
                className="markdown-container",
                children=[
                    html.Div(
                        className="close-container",
                        children=html.Button(
                            "Close",
                            id="markdown_close",
                            n_clicks=0,
                            className="closeButton",
                            style={"color": "DarkBlue"},
                        ),
                    ),
                    html.Div(
                        className="markdown-text", children=dcc.Markdown(readme_md)
                    ),
                ],
            )
        ],
    )


# ======= Callback for modal popup =======
@app.callback(
    Output("markdown", "style"),
    [Input("learn-more-button", "n_clicks"), Input("markdown_close", "n_clicks")],
)
def update_click_output(button_click, close_click):
    if button_click > close_click:
        return {"display": "block"}
    else:
        return {"display": "none"}


def description_card():
    """
    :return: A Div containing dashboard title & descriptions.
    """
    return html.Div(
        id="description-card",
        children=[
            html.Br(),
            html.H5("Explainable AI"),
            html.H3("Welcome to our Dashboard"),
            html.Div("by Niklas Groiss, Markus Schleicher and Leonard Uhlisch", style={'weight': 'bold'}),
            html.Div(
                id="intro",
                children='This is our interactive approach for the education of explainable AI (XAI) using a Grad-CAM on images. Have a look at "Learn more" to find out about all the functions.',
            ),
        ],
    )


def generate_control_card():
    """
    :return: A Div containing controls for graphs.
    """
    return html.Div(
        id="control-card",
        children=[

            html.Br(),

            html.P("Select Architecture", id="select-model-markdown"),
            dcc.Dropdown(
                id="model-select",
                options=model_labels,
                value=model_labels[0]['value'],
                clearable=False,
            ),
            dbc.Tooltip(
                "Here you can select the architecture on which the Grad-CAM will be based later. You can get more information about the selected architecture by clicking on LEARN MORE in the top right corner.",
                target="select-model-markdown",
                style={
                    'font-size': 18,
                    'maxWidth': 600,
                    'width': 300,
                    'background-color': 'white',
                    'borderStyle': 'none',
                    "color": "black"}
            ),
            html.Br(),

            html.Div(
                id="reset-btn-outer",
                children=html.Button(id="reset-btn", children="Reset", n_clicks=0),
            ),
            html.Br(),

            html.Div([
                dcc.Upload(
                    id='upload-image',
                    children=html.Div([
                        'Drag and Drop or ',
                        html.A('Select Files')
                    ]),
                    style={
                        'width': '100%',
                        'height': '100px',
                        'lineHeight': '100px',
                        # 'borderStyle': 'dashed',
                        # 'borderRadius': '5px',
                        # 'background-color': '#88888f',
                        'textAlign': 'center',
                        # 'margin': '10px'

                    },
                    # Don't allow multiple files to be uploaded
                    multiple=False
                ),
            ]),
        ],
    )


 


def generate_img_card():
    return html.Div(id='output-image-upload'),


def configuration_card():
    return html.Div(
        id="config-card",
        children=[
            slider_alpha(),
            html.Br(),
            html.P("Select heatmap-style", id="select-heatmap-markdown"),
            dcc.Dropdown(
                id="heatmap-select",
                options=heatmap_labels,
                value=heatmap_labels[0]['value'],
                clearable=False,
            ),
            dbc.Tooltip(
                "The color scheme of the heatmap can be defined here.",
                target="select-heatmap-markdown",
                style={
                    'font-size': 18,
                    'maxWidth': 600,
                    'width': 300,
                    'background-color': 'white',
                    'borderStyle': 'none',
                    "color": "black"},
            ),
            html.Br(),

            html.P("Select Grad-CAM", id="select-gradcam-markdown"),
            dcc.Dropdown(
                id="gradcam-select",
                options=gradcam_labels,
                value=gradcam_labels[0]['value'],
                clearable=False,
            ),
            dbc.Tooltip(
                "The Grad-CAM variant can be selected here. You can find more information about the implemented Grad-CAM variants under LEARN MORE.",
                target="select-gradcam-markdown",
                style={
                    'font-size': 18,
                    'maxWidth': 600,
                    'width': 300,
                    'background-color': 'white',
                    'borderSTyle': 'none',
                    "color": 'black'},
            ),
        ]
    )


def slider_alpha():
    """
    :return: A Div containing slider.
    """
    return html.Div(
        [html.Br(),
         dcc.Slider(
             id='slider-alpha',
             min=0,
             max=100,
             step=1,
             value=80,  # start value
             tooltip={'always_visible':False},  
         ),
         html.Div(id='slider-output-container')
         ]
    )

def slider_predictions():
    """
    :return: A Div containing slider.
    """
    return html.Div(
        [
            dcc.Slider(
                id='slider-predictions',
                min=1,
                max=10,
                step=1,
                value=3,  # start value
                tooltip={'always_visible':False},
            ),
            html.Div(id='slider_predictions-output-container')
        ]
    )




app.layout = html.Div(
    id="app-container",
    children=[
        # Banner
        html.Div(
            id="banner",
            className="banner",
            children=[
                html.Img(src="https://www.hs-aalen.de/assets/frontend/logo-cd36bda6c609c67ce7cab4412f277190.png"),
                html.Div(
                    html.Button(
                        "Learn more",
                        id="learn-more-button",
                        n_clicks=0,
                        style={"width": "auto", }
                    ),
                ),
                # Adding the modal content here. It is only shown if the show-modal
                # button is pressed
                make_modal(),
            ],
        ),

        html.Div(id='top-bar', className='row'),
        html.Div(className='row', style={'height': '10px'}),
        # left column
        html.Div(
            id="left-column",
            className="four columns",
            children=[
                html.Div(
                "1",
                id="step-1",
                style={
                    "textAlign": "center",
                },
                className="number-circle"),
                description_card(),
                generate_control_card()]
            +   [
                    html.Div(
                        ["initial child"], id="output-clientside", style={"display": "none"}
                    )
            ],
        ),
        # middle column
        html.Div(
            id="middle-column",
            className="four columns",
            children=[
                html.Div(
                "2",
                id="step-2",
                style={
                    "textAlign": "center",
                },
                className="number-circle"),
                html.Br(),
                html.Div(id='original-image-upload'),
                html.Br(),
                slider_predictions(),
                html.Br(),
                # dcc.Loading(
                #     id="loading-1",
                #     type="default",
                #     children=html.Div(id="loading-output-1")
                # ),
                html.Div(id="prediction-table"),
                dbc.Tooltip(
                    "The predictions based on the uploaded image are displayed here. The number of predictions can be limited using the slider.",
                target="slider_predictions-output-container",
                style={
                    'font-size': 18,
                    'maxWidth': 600,
                    'width': 300,
                    'background-color':'white',
                    'borderStyle': 'none',
                    "color": "black"}
                )
                # html.Div(id='image'),

            ],
        ),
        # right column
        html.Div(
            id="right-column",
            className="four columns",
            children=[
                html.Div(
                "3",
                id="step-3",
                style={
                    "textAlign":"center",
                },
                className="number-circle"),
                html.Br(),
                html.Div(id='grad-cam-image'),
                configuration_card(),

            ],
        ),
    ],
)



########## middle column updates ################
# load image
@app.callback(Output('original-image-upload', 'children'),
              [
                  Input('reset-btn', 'n_clicks'),
                  Input('upload-image', 'contents'), 
              ],
              State('upload-image', 'filename'),
              )
def update_original_img_card(reset_click,contents, filename):
    reset = False

    ctx = dash.callback_context

    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
        #print("prop_id", prop_id)
        if prop_id == "reset-btn":
            reset = True
            #effect on layout
            #image delete, prediction delete
            return dbc.Card([
                dbc.CardHeader("Selected image"),
                dbc.CardBody(html.P("No image selected", className="card-text")),
                dbc.CardFooter(""),

                ],
                style={'width': '100%','object-fit': 'contain'}
                )
    if contents is not None:
        return dbc.Card(
            [
                dbc.CardHeader("Selected image"),
                dbc.CardImg(src=contents, bottom=True),
                dbc.CardFooter(html.P(filename, className="card-text")),
            ],
            style={'width': '100%', 
                'object-fit': 'contain'}
        )
    else:
        return dbc.Card([
                dbc.CardHeader("Selected image"),
                dbc.CardBody(html.P("No image selected", className="card-text")),
                dbc.CardFooter(""),
                ],
                style={'width': '100%','object-fit': 'contain'}
                )


# update slider
@app.callback(
    Output('slider_predictions-output-container', 'children'),
    [Input('slider-predictions', 'value'),
     Input('upload-image', 'contents'),],
    )
def update_header_predictions(value, contents):
    # if contents is not None:
    return 'Predictions Top: {} of 10'.format(value)
    # else:
    #     return 'No image to predict.'


# update prediction
@app.callback(
    Output("prediction-table", "children"),
    [
        Input("slider-predictions", "value"),
        Input('model-select', 'value'),
        Input('upload-image', 'contents'),
        Input("reset-btn", "n_clicks"),
    ],
)
def update_prediction_output(value, value_model, contents, reset_click):
    reset = False

    ctx = dash.callback_context


    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
        #print("prop_id", prop_id)
        if prop_id == "reset-btn":
            reset = True
            #no prediction table
            return html.Div()            
    
    df_pred= pd.DataFrame()

    if contents is not None:
        #print(contents[23:-2])
        #print(type(contents))
        #print('update_predction_output', value_model)
        img = base64.b64decode(contents[23:])
        img = Image.open(io.BytesIO(img))  
        
        if value_model == "xception":
            img = img.resize((299, 299))
        else:
            img= img.resize((224, 224))

        #print("IMAGE", type(img))
        table_data = models.get_predictions(img, value_model)
        table_df = table_data.iloc[:value,]
        df_pred = table_df.iloc[:value, [1,2]] # show only columns 1 and 2 in predictions
        df_pred.style.format({'Probability': "{:.2f%}"})

        return html.Div(
            [
                DataTable(
                    id="prediction-df",
                    export_format='csv',
                    data=df_pred.to_dict("records"),
                    columns=[
                        dict(id='Predicted Object', name='Predicted Object'),
                        dict(id='Probability', name='Probability', type='numeric', format=FormatTemplate.percentage(4))
                        #"name": str(i), "id": str(i)} for i in df_pred.columns
                    ],
                )
            ]
        )
    else:
        return html.Div()   


#### updates right column #############


@app.callback(Output('grad-cam-image', 'children'), 
            [
                Input('reset-btn', 'n_clicks'),
                Input('heatmap-select', 'value'),
                Input('gradcam-select','value'),
                Input('upload-image', 'contents'),
                Input('slider-alpha', 'value'),
                Input('model-select', 'value'),
            ], 
            State('upload-image', 'filename'), 
            State('prediction-table', 'children'),#state of predicition
            )
def update_gradcam_img_card(reset_click, value_heatmap, value_gradcam, contents, value_alpha, model_name, filename, prediction_table):

    reset = False

    ctx = dash.callback_context

    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
        #print("prop_id", prop_id)
        if prop_id == "reset-btn":
            reset = True
            #effect on layout: image delete
            return dbc.Card([
                dbc.CardHeader("GradCAM image"),
                dbc.CardBody(html.P("No image selected", className="card-text")),
                dbc.CardFooter(""),
                ],
                style={'width': '100%','object-fit': 'contain'}
                )
    
    if contents is not None:
        # print(value, contents, "\n ###############", filename)
        value_alpha = value_alpha/100
        img_to_pil = base64.b64decode(contents[23:])
        img_to_pil = Image.open(io.BytesIO(img_to_pil))
        
        #save width, height for resizing
        width, height = img_to_pil.size
        if model_name == "xception":
            img_to_pil = img_to_pil.resize((299, 299))
        else:
            img_to_pil = img_to_pil.resize((224, 224))
        #print('Shape', img_to_pil.size) #4968, 2794
        #img_to_pil = img_to_pil.resize((299, 299))
        #print(type(value_heatmap))
        #print(type(value_gradcam))
        #print(contents)
        #print(value_alpha)
        gradcam_img = models.get_gradcam_img(img_to_pil, model_name, value_heatmap, value_gradcam,  value_alpha)
        #gradcam_img = str(gradcam_img)
        #print("test 1", gradcam_img[:10])
        #gradcam_img = gradcam_img[2:]
        #print("test2", gradcam_img[:10])
        #print(type(gradcam_img))
        # apply gradcam to img
        #print('Shape-small', gradcam_img.size)
        gradcam_img=gradcam_img.resize((width,height))

        
        #print('Shape-after', gradcam_img.size)

       
        enc_format=filename
        #print(enc_format)

        enc_format='png'

      


         # plot gradcam in card
        footer_text=filename
        if value_gradcam == "grad_cam":
            footer_text = 'This has caused the prediction.' 
        elif value_gradcam == "c_expl_heatmap":
            footer_text = 'This has caused distraction from predicting.'

        return dbc.Card(
            [
                dbc.CardHeader("GradCAM image"),
                dbc.CardImg(src=f"data:image/{enc_format};base64, " + models.pil_to_b64(gradcam_img)),
                dbc.CardFooter(html.P(footer_text, className="card-text")),
            ],
            style={'width': '100%',
                'object-fit': 'contain'}
        )

    else:
        return dbc.Card(
            [
                dbc.CardHeader("GradCAM image"),
                dbc.CardBody(html.P("No image selected", className="card-text")),
                #dbc.CardImg(src=contents, bottom=True),
                dbc.CardFooter(""),
            ],
            style={'width': '100%','object-fit': 'contain'}
        )

@app.callback(
    Output('slider-output-container', 'children'),
    [Input('slider-alpha', 'value')])
def update_output(value):
    return 'Defined Alpha: {} %'.format(value)



import plotly.graph_objs as go

#not used
def original_graph(contents):
    return dcc.Graph(
        id="interactive-image",
        figure={
            "data": [],
            "layout": {
                #"autosize": True,
                #"paper_bgcolor": "#272a31",
                #"plot_bgcolor": "#272a31",
                "margin": go.Margin(l=40, b=40, t=26, r=10),
                "xaxis": {
                    "range": (0, 1527),
                    "scaleanchor": "y",
                    "scaleratio": 1,
                    "color": "white",
                    "gridcolor": "#43454a",
                    "tickwidth": 1,
                    "showgrid" :False,
                },
                "yaxis": {
                    "range": (0, 1200),
                    "color": "white",
                    "gridcolor": "#43454a",
                    "tickwidth": 1,
                },
                "images": [
                    {
                        "xref": "x",
                        "yref": "y",
                        "x": 0,
                        "y": 0,
                        "yanchor": "bottom",
                        "sizing": "stretch",
                        "sizex": 1527,
                        "sizey": 1200,
                        "layer": "below",
                        "source": contents,
                    }
                ],
                "dragmode": "select",
            },
        },
)


# Run the server
if __name__ == "__main__":
    app.run_server(debug=False)
