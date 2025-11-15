# Use an official python image.
FROM python:3.9

# Copy the requirements file into the container
COPY ./requirements.txt /deploy/

# Install the Python packages required for our model and framework
RUN python -m pip install --upgrade pip
RUN python -m pip install -r /deploy/requirements.txt

# Copy our application script into the container at the required location
COPY ./predictor.py /deploy/

# Set the entrypoint to gunicorn
WORKDIR /deploy
ENV PORT 8080
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 predictor:app
