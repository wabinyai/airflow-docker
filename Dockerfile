# Use the official Python image as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Set environment variables for Airflow
ENV AIRFLOW_HOME=/app/airflow

# Initialize Airflow database
RUN airflow db init

# Expose the port for the Airflow webserver
EXPOSE 8080

# Set the default command to start the Airflow webserver
CMD ["airflow", "webserver"]