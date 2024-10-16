# FROM python:3.11.5
#
# # Install system dependencies for OpenCV
# RUN apt-get update && apt-get install -y \
#     libgl1-mesa-glx \
#     libglib2.0-0 \
#     && rm -rf /var/lib/apt/lists/*
#
# # Set the working directory inside the container
# WORKDIR /code
#
# # Copy the requirements file into the container
# COPY ./requirements.txt /code/requirements.txt
#
# # Install dependencies
# RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
#
# # Copy your application files to the working directory
# COPY ./fast-api-backend /code/fast-api-backend
#
# # Expose ports for both FastAPI and Dash
# EXPOSE 8000
# EXPOSE 8051
#
# # Command to run both FastAPI and Dash
# CMD ["uvicorn", "fast-api-backend.main:app", "--host", "0.0.0.0", "--port", "8000"]



FROM python:3.11.5

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY main.py /code/main.py


EXPOSE 8000
# EXPOSE 8051

# ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

CMD bash -c "uvicorn main:app --host 0.0.0.0 --port 8000"
#& python /code/plotly-dash-frontend/index.py"
