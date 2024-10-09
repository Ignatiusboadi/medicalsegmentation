import uuid
import dash
import requests
from dash import dcc, html, Input, Output, State, callback, dcc, ctx
from app import api_url
import dash_bootstrap_components as dbc
import os
import zipfile
import io
import base64
import shutil

from dash.exceptions import PreventUpdate
from flask import send_file


def create_zip(file_data_list, filenames):
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w') as zf:
        for data, name in zip(file_data_list, filenames):
            decoded_data = base64.b64decode(data.split(',')[1])
            zf.writestr(name, decoded_data)
    memory_file.seek(0)
    return memory_file


upload_message = 'Multiple file uploads are allowed. Please place all scans in a folder, select all files, and upload them together.'
UPLOAD_DIRECTORY = 'uploads'

layout = html.Div(style={'padding-top': '30px', 'background-image': 'url("/assets/brain_imag_bg.webp"',
                         'height': '100vh'}, children=[
    dbc.Row(children=[
        dbc.Col(children=[
            html.H2("Brain Image Tumor Segmentation", className="text-center mb-4",
                    style={'textAlign': 'center', 'font-weight': 'bold', 'color': '#3B1C0A', 'padding-top': '10px',
                           'font-size': '200%'}),
        ], width=11),
        dbc.Col(dbc.Button(id='logout', children='Logout', n_clicks=None))], justify='center'),
    dbc.Container(style={'padding-top': '100px'}, children=[
        dbc.Row(children=[
            dbc.Col(children=[
                dbc.Card([
                    dbc.CardBody(style={'background-color': 'GhostWhite'}, children=[
                        dbc.Label("Upload Folder", html_for="upload-files"),
                        dbc.Card([
                            dbc.CardBody(style={'textAlign': 'center', }, children=[
                                dcc.Upload(id='upload-files',
                                           children=html.Div([
                                               'Drag and Drop or ', html.A('Select Files')]),
                                           multiple=True,
                                           style={'width': '100%', 'height': '60px',
                                                  'lineHeight': '60px', 'borderWidth': '1px',
                                                  'borderStyle': 'dashed', 'borderRadius': '5px',
                                                  'textAlign': 'center',
                                                  'font-family': 'Lucida Console'}), ])]),
                        html.Br(),
                        html.Em(upload_message, style={'color': 'green'}),
                        html.Br(),
                        dbc.Row(children=[dbc.Col(children=[
                            dbc.Button("Segment Scans", id='segment', color="danger", className='text-center',
                                       outline=True, size='md',
                                       style={'padding-left': '45px', 'padding-right': '45px'}),
                            dcc.Loading(dcc.Download(id='download-btn'), fullscreen=True, )
                        ],
                            width={'offset': 4}, style={'padding-left': '25px', 'padding-right': '25px'})],
                            justify="center"),
                    ], ),
                ], ),
            ], width={'size': 6, 'offset': 3})], style={'padding-bottom': '50px'}),
    ], fluid=True)])


@callback(Output('url', 'pathname'),
          Output('token', 'data'),
          Output('logout', 'n_clicks'),
          Input('logout', 'n_clicks'))
def log_out(n_clicks):
    if n_clicks is None:
        raise PreventUpdate
    if n_clicks:
        return '/', None, None


@callback(Output('download-btn', 'data'),
          Output('segment', 'n_clicks'),
          Input('upload-files', 'filename'),
          Input('upload-files', 'contents'),
          Input('token', 'data'),
          Input('segment', 'n_clicks'))
def segment_images(file_names, file_contents, bearer_token, n_clicks):
    if not n_clicks:
        raise PreventUpdate
    headers = {
        'Authorization': f"Bearer {bearer_token}"
    }
    segment_api = f"{api_url}/prediction"
    if n_clicks > 0 and file_contents is not None:
        folder_name = str(uuid.uuid4())
        folder_path = os.path.join(UPLOAD_DIRECTORY, folder_name)

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        for i, content in enumerate(file_contents):
            content_type, content_string = content.split(',')
            decoded_file = base64.b64decode(content_string)

            with open(os.path.join(folder_path, file_names[i]), 'wb') as f:
                f.write(decoded_file)

        zip_filename = f"{folder_name}.zip"
        zip_filepath = os.path.join(UPLOAD_DIRECTORY, zip_filename)

        with zipfile.ZipFile(zip_filename, 'w') as zf:
            for file_name in file_names:
                zf.write(os.path.join(folder_path, file_name), file_name)
        shutil.rmtree(folder_path)

        try:
            with open(zip_filename, "rb") as zip_file:
                files = {
                    "file": (zip_filename, zip_file, "application/zip")
                }
                response = requests.post(segment_api, headers=headers, files=files)
                print('about to remove')
                os.remove(zip_filename)
                print('removed')
        except FileNotFoundError:
            return f"Error: The file {zip_filename} does not exist."
    file_content = response.content
    print(response.headers)
    content_disp = response.headers.get('Content-Disposition')
    file_name = content_disp.split('filename=')[1].strip('"')

    return dcc.send_bytes(file_content, file_name), 0


# if __name__ == '__main__':
#     app.run_server(debug=True, port=8051)
