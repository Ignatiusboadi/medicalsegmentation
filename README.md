## Medical Image Segmentation

In this project, we will focus on MLOps by developing an image segmentation system for CT scans and brain MRIs specifically aimed at detecting brain tumors. The goal of this project is to assist medical practitioners in easily identifying brain tumors, including their size and location within the brain. With our system, doctors will simply need to upload the scans, and they will receive an output image highlighting the areas affected by the tumors.

## User Instructions
1. Clone the GitHub repository
-     git clone https://github.com/Ignatiusboadi/medicalsegmentation.git
      cd fmedicalsegmentation

2. Create a virtual environment
-     virtualenv venv
      source venv/bin/activate

3. Install the required packages
-     pip install -r requirements.txt

4. To run the backend locally, run the folllowing line of code
-     uvicorn main:app --reload

## License

This project is licensed under the [MIT License](LICENSE.md). Please read the License file for more information.
<!-- 
### Docker
- Ensure you have Docker installed, you can install it from [here](https://docs.docker.com/desktop/?_gl=1*wtu5yy*_gcl_au*MTcwMDA1NDUzMi4xNzI4MTI3ODE0*_ga*MzI4MDQwOTk1LjE3MjcyODA5OTg.*_ga_XJWPQMJYHQ*MTcyODEyNzc4Ny4zLjEuMTcyODEyNzgxNC4zMy4wLjA.).
  * Build the docker image and run the docker container.
    
    docker build -t medical-segmentation-system .  (copy and paste upto the dot)
    
    docker run -d -p 8000:8000 medical-segmentation-system
  * Open (http://localhost:8000) on your browser to access the FastAPI application.
 ### Data Versioning
 - Data versioning with DVC
   * Install [DVC](https://dvc.org/doc/install), if not yet installed.
  
     dvc init
   * Track data files assuming your dataset is in a folder named data.
     
     dvc add data/
   * Push data to a remote storage
     
     dvc remote add -d myremote <remote_storage_url>  #the remote storage could be AWS, GCP, Azure etc
     
     dvc push
 ### Frontend with Plotly Dash
- It provides an easy way to create a frontend interface for users to interact with the brain tumour segmentation service.
- We chose Plotly Dash as it is a powerful framework that's well-suited for data visualizations given that we will be using image data.
  * Create a plotly dash script, e.g app.py
  * Run the Plotly Dash application
    
    pip install dash
    
    python app.py -->
