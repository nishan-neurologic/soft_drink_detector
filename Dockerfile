# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory
WORKDIR /app

# Install required system packages
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Copy the requirements file into the container
COPY requirements.txt /app

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Install PyTorch and torchvision for CPU
RUN pip install torch torchvision -f https://download.pytorch.org/whl/cpu/torch_stable.html
RUN python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Copy the rest of the application code
COPY . /app

# Make port 5000 available to the world outside this container
EXPOSE 5000
WORKDIR /app/src
# Define environment variable
ENV FLASK_APP=app.py

# Run the application when the container launches
CMD ["flask", "run", "--host=0.0.0.0"]
