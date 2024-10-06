from brain_dataset import BrainDataset
from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, status, Header
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse
from google.cloud import storage
from datetime import datetime, timedelta, timezone
from typing import Union
from jwt import PyJWTError
from passlib.context import CryptContext
from torch.utils.data import DataLoader
from torchvision import transforms

import cv2
import faker
import os
import zipfile
import json
import jwt
import numpy as np
import pandas as pd
import shutil
import segmentation_models_pytorch as smp
import torch
import warnings
import yagmail
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.metrics.base_metric import generate_column_metrics
from evidently.metrics import *
from evidently.test_suite import TestSuite
from evidently.tests.base_test import generate_column_tests
from evidently.test_preset import DataStabilityTestPreset, NoTargetPerformanceTestPreset, RegressionTestPreset
from evidently.tests import *
import shutil
from evidently import ColumnMapping

warnings.filterwarnings("ignore", category=RuntimeWarning)

cred = 'focus-surfer-435213-g6.json'

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = cred


def clamp_tensor(x):
    return x.clamp(0, 1)


def upload_to_gcp(source_file_name, destination_folder):
    bucket_name = 'face-verification-images'
    destination_blob_name = f'{destination_folder}/{source_file_name}'
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(f"File {source_file_name} uploaded to {destination_blob_name}.")

########################################################################################
# TOKEN AUTHENTICATION
########################################################################################
SECRET_KEY = "fdb3e44ba75f4d770ee8de98e488bc3ebcf64dc3066c8140a1ae620c30964454"  # Replace with your own secret key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

users_db = {
    "admin": {"username": "admin", "password": pwd_context.hash("adminpass"), "role": "admin",
              'email': 'ignatiusboadi@gmail.com'},
    "user": {"username": "user", "password": pwd_context.hash("userpass"), "role": "user",
             'email': 'iboadi@aimsammi.org'},
    "leo": {"username": "leo", "password": pwd_context.hash("l001"), "role": "user", "email": "lsanya@aimsammi.org"},
    "milli": {"username": "milli", "password": pwd_context.hash("m001"), "role": "user",
              "email": 'momondi@aimsammi.org'}
}

upload_dir = 'uploads'
root_dir = '/'
os.makedirs(upload_dir, exist_ok=True)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def authenticate_user(username: str, password: str):
    user = users_db.get(username)
    if not user or not verify_password(password, user["password"]):
        return False
    return user


def create_access_token(data: dict, expires_delta: Union[timedelta, None] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
        return username
    except PyJWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


app = FastAPI()


@app.get("/")
def index():
    return {"message": "Welcome to the Image Segmentation using FastAPI app!"}


@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    email = user['email']
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]}, expires_delta=access_token_expires
    )

    yag = yagmail.SMTP('ammi.mlops.group1@gmail.com', 'pktwlpqogrkotiyg')
    message = f'''Dear User,
                        Kindly find below the bearer token for you to access
                        {access_token}

                        Kindest regards,
                        Group 1'''
    yag.send(email, f'Bearer token', message)
    return {"access_token": access_token, "token_type": "bearer"}


#######################################################################################
#DATA DRIFT DETECTION
#######################################################################################
@app.post("/Drift Monitoring")
async def Data_Drift_and_Test(token: str = Depends(oauth2_scheme)):

    decode_token(token)
    train_json = 'train_annotations.coco.json'
    test_json = 'test_annotations.coco.json'

    with open(train_json, 'r') as f:
        train_data = json.load(f)
    with open(test_json, 'r') as f:
        test_data = json.load(f)

    train_images = pd.DataFrame(train_data['images'])
    test_images = pd.DataFrame(test_data['images'])

    column_mapping = ColumnMapping(
        target=None,
        categorical_features=[],
        numerical_features=['width', 'height']
    )

    report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset(),
        ColumnSummaryMetric(column_name='height'),
        generate_column_metrics(ColumnQuantileMetric, parameters={'quantile': 0.25}, columns=['id']),
        ColumnDriftMetric(column_name='width')
    ])
    report.run(reference_data=train_images, current_data=test_images)

    html_report_path = os.path.join(upload_dir, 'data_drift_report.html')
    report.save_html(html_report_path)

    tests = TestSuite(tests=[
        TestNumberOfColumnsWithMissingValues(),
        TestNumberOfRowsWithMissingValues(),
        TestNumberOfConstantColumns(),
        TestNumberOfDuplicatedRows(),
        TestNumberOfDuplicatedColumns(),
        TestColumnsType(),
        TestNumberOfDriftedColumns(),
    ])

    tests.run(reference_data=train_images, current_data=test_images)
    html_tests_path = os.path.join(upload_dir, 'data_tests.html')
    tests.save_html(html_tests_path)

    zip_file_path = os.path.join(upload_dir, 'report_bundle.zip')
    with zipfile.ZipFile(zip_file_path, 'w') as zipf:
        zipf.write(html_report_path, os.path.basename(html_report_path))
        zipf.write(html_tests_path, os.path.basename(html_tests_path))

    return FileResponse(zip_file_path, media_type='application/zip', filename='report_bundle.zip')


#######################################################################################
# PREDICTION ENDPOINT
#######################################################################################
@app.post("/prediction")
async def image_segmentation(file: UploadFile = File(...), token: str = Depends(oauth2_scheme)):
    decode_token(token)
    f = faker.Faker()
    folder_name = f.name().split()[0]
    temp_zip_path = f"{folder_name}.zip"

    with open(temp_zip_path, "wb") as temp_zip_file:
        content = await file.read()
        temp_zip_file.write(content)

    with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
        zip_ref.extractall(folder_name)
    os.remove(temp_zip_path)

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229]),
        transforms.Lambda(clamp_tensor)
    ])
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = torch.load('models/best_model.pth')
    model.eval()
    model = model.to(device)

    output_dir = f"{folder_name}_output"
    os.mkdir(output_dir)
    data2predict = BrainDataset(root_dir, folder_name, transform=transform)
    pred_loader = DataLoader(data2predict, batch_size=1, shuffle=False)

    for i, (data, filename) in enumerate(pred_loader):
        data = data.to(device)
        pred_logits = model(data)
        pred_binary = (pred_logits > 0.5).float()
        mask = pred_binary.squeeze().cpu().numpy()
        original_size = data2predict.get_original_size(i)
        mask_resized = cv2.resize(mask, (original_size[1], original_size[0]))

        mask_filename = os.path.basename(filename[0])
        output_mask_path = os.path.join(output_dir, mask_filename)

        mask_resized = (mask_resized * 255).astype(np.uint8)
        cv2.imwrite(output_mask_path, mask_resized)
    output_zip = f'segmented_{''.join(file.filename.split('.')[:-1])}_images.zip'
    shutil.make_archive(output_dir, output_zip)

    zip_filename = output_zip
    return FileResponse(zip_filename, media_type='application/zip', filename=zip_filename)
