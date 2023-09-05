# Use an official Python runtime as a parent image
FROM python:3.10.6-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements_prod.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Run Uvicorn with FastAPI application
# CMD ["uvicorn", "api.fast:app", "--host", "0.0.0.0", "--port", "8000"]
CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
