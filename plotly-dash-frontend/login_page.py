from dash import dcc, html, Input, Output, State, callback
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import requests
import datetime

email_n = "Once the loading screen finishes, please check your email for a token. Enter the token below and click 'Authenticate'."

# Layout for login page
layout = html.Div(style={'background-color': 'GhostWhite', 'height': '100vh', 'padding-bottom': '100px',
                         'padding-top': '50px', 'background-image': 'url("/assets/brain_imag_bg.webp"'},
                  children=[
                      dbc.Row(children=[
                          dbc.Col(children=[
                              html.H2('BRAIN IMAGE TUMOR SEGMENTATION PORTAL',
                                      style={'textAlign': 'center', 'font-weight': 'bold', 'color': 'red',
                                             'font-size': '200%'}), ], )], justify='center'),
                      dbc.Container(style={'padding-top': '100px'}, children=[
                          dbc.Row(children=[
                              dbc.Col(html.H2("Login", className="text-center mb-4"), width=12)]),
                          dbc.Row(children=[
                              dbc.Col(children=[
                                  dbc.Card([
                                      dbc.CardBody(style={'background-color': 'GhostWhite'}, children=[
                                          dbc.Label("Username:", html_for="username-input"),
                                          dcc.Input(id='username-input', type='text', placeholder='Enter Username',
                                                    className="form-control mb-3"),
                                          dbc.Label("Password:", html_for="password-input"),
                                          dcc.Input(id='password-input', type='password', placeholder='Enter Password',
                                                    className="form-control mb-3"),
                                          html.Br(),
                                          dbc.Row(children=[dbc.Col(children=[
                                              dbc.Button("Generate Token", id='token-btn', color="primary",
                                                         className='text-center', outline=True,
                                                         size='md', style={'padding-left': '45px',
                                                                           'padding-right': '45px'}), ],
                                              width={'offset': 3},
                                              style={'padding-left': '25px', 'padding-right': '25px'})],
                                              justify="center"),
                                          dcc.Loading(html.Em(email_n, id='auth-output',
                                                              style={'color': 'green', 'font-size': '13px'}),
                                                      type='default', fullscreen=True, ),
                                          dbc.Label("Token:", html_for="token-input"),
                                          dcc.Input(id='token-input', type='text', placeholder='Enter Token',
                                                    className="form-control mb-3"),
                                          html.Br(),
                                          dbc.Row([
                                              dbc.Col(
                                                  dbc.Button("Authenticate", id='auth-btn', color="success",
                                                             outline=True, className='mt-1', size='md',
                                                             style={'padding-left': '60px', 'padding-right': '60px'}),
                                                  width={'offset': 3},
                                                  style={'padding-left': '35px', 'padding-right': '35px'})],
                                              justify="center"), ])])], width=4)], justify="center"),
                          dbc.Row([
                              dbc.Col(html.P(id='output-message', className="mt-4 text-center",
                                             style={'color': 'red', 'text-weight': 'bold'}), )])
                      ], fluid=True)])


@callback(
    Output('auth-output', 'children'),
    Output('token', 'data', allow_duplicate=True),
    Input('token-btn', 'n_clicks'),
    State('username-input', 'value'),
    State('password-input', 'value'),
    config_prevent_initial_callbacks=True
)
def generate_token(n_clicks, username, password):
    if not n_clicks or username is None or password is None:
        raise PreventUpdate
    api_url = 'http://127.0.0.1:8000'
    token_url = f'{api_url}/token'

    auth_data = {
        'username': username,
        'password': password
    }

    token_response = requests.post(token_url, data=auth_data)
    print(token_response.status_code)
    access_token = token_response.json().get('access_token')

    if token_response.status_code == 200:
        return 'Kindly check your email for a token, enter it below and click Authenticate.', access_token,
    return 'Wrong Username or Password', access_token


@callback(Output('url', 'pathname', allow_duplicate=True),
          Input('token', 'data'),
          Input('auth-btn', 'n_clicks'),
          State('token-input', 'value'),
          config_prevent_initial_callbacks=True)
def authenticate_user(syst_token, n_clicks, user_token):
    print('authbtn', datetime.datetime.now(), n_clicks)
    # if n_clicks is None:
    #     raise PreventUpdate
    if not n_clicks:
        return '/'
    print('store', syst_token)
    print('input', user_token)
    if syst_token is not None and user_token is not None and syst_token == user_token and n_clicks:
        return '/main'
    elif syst_token != user_token and n_clicks:
        return '/'
