# Use an official Python runtime as a parent image
FROM python:3.6

# Add the current directory into the path /demo in the container
ADD . /demo

# Set the working directory to /demo
WORKDIR /demo

# Install required packages available from apt-get
RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get install -y gfortran openmpi-bin libopenmpi-dev libblas-dev liblapack-dev

# Install required Python packages using pip
RUN pip install numpy mpi4py openmdao
RUN pip install petsc==3.10
RUN pip install petsc4py==3.10

# Make port 80 available to the world outside this container
# EXPOSE 80

# Run run.py when the container launches
CMD ["bash", "mpirun --allow-run-as-root -np 10", "python", "problem.py", "10"]
