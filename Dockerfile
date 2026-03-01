# syntax=docker/dockerfile:1
#
# Dockerfile for building and running the generative AI project.  The
# container installs all Python dependencies defined in the accompanying
# requirements.txt file and sets the working directory to the root of
# the project.  A default command is provided to run the package as a
# module, which will dispatch to the appropriate entry point defined in
# your code.

FROM python:3.11-slim

# Create and set the working directory
WORKDIR /app

# Copy the project into the container
COPY . /app

# Upgrade pip and install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Set the default command; override this in docker-compose.yml if needed
CMD ["python", "-m", "generative_ai_project"]