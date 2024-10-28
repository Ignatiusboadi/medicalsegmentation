FROM python:3.11.5

# Install dependencies needed for certain libraries (optional, based on your project)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /code

# Copy requirements.txt and install dependencies
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the rest of the application code to the container
COPY ./ /code/

# Expose port 8000 for FastAPI
EXPOSE 8000

# Command to run the FastAPI app using Uvicorn on port 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

