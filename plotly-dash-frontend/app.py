import dash
import dash_bootstrap_components.themes as themes

api_url = 'http://127.0.0.1:8000'

FONT_AWESOME = (
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css"
)

app = dash.Dash(__name__, title='IMAGE SEGMENTATION', external_stylesheets=[themes.SIMPLEX, FONT_AWESOME],
                meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}], update_title=None,
                suppress_callback_exceptions=True, assets_folder='assets', assets_url_path='/assets/')

server = app.server