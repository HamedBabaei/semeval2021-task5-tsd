import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
from ast import literal_eval
from dashhtmlparser import DashHTMLParser

app = dash.Dash(__name__)

app.layout = html.Div(style={'alignItems':'center', 'backgroundColor':'#F8F8FF'}, children=[
    html.Div(className='break'),
    dcc.Input(id='my-id', value='Dash App', type='file', className='input'),
    html.Div(className='break'),
    html.Div(id='my-div')
])

@app.callback(
    Output(component_id='my-div', component_property='children'),
    [Input(component_id='my-id', component_property='value')]
)
def update_output_div(input_value):
    """
        Loaing Dataset and create a html for Dash
    Arguments:
       input_value(string): path to input dataset
    Returns:
       html(Dast)
    """
    file = str(input_value).split("C:\\fakepath\\")[-1]
    trial = pd.read_csv(file)
    trial["spans"] = trial.spans.apply(literal_eval)
    _html = [html_to_dash(display_toxics(trial.spans[index], trial.text[index])) 
             for index in range(0, trial.shape[0])]
    return html.P(_html)

def html_to_dash(html_string):
    """
       Creating Dash HTML from string htmls using DashHTMLParser
    Arguments:
       html_string
    Returns:
       dash_object
    """
    parser = DashHTMLParser()
    parser.feed(html_string)
    return parser.dash_object

def display_toxics(span, text):
    """
       Highliting toxic part of text as a red, and adding span class to comment
    Arguments:
       span(list): list of toxic word/phrases character place
       text(string): a comment
    returns:
       html: returns "<p class='spans'> coment highlited toxic with red color </p>"
    """
    html = "<p class='spans'>"
    for ind, char in enumerate(text):
        if ind in span:
            html += "<b style='color:red'>" + char + '</b>'
        else:
            html += char
    html += '</p>'
    return html


if __name__ == '__main__':
    app.run_server(debug=True)