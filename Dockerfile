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
