# The Google Cloud Platform Python runtime is based on Debian Jessie
# You can read more about the runtime at:
#   https://github.com/GoogleCloudPlatform/python-runtime
FROM gcr.io/google_appengine/python

RUN apt-get update
RUN apt-get -y install python3 python3-pip python3-dev build-essential

COPY requirements.txt /app/
RUN pip3 install --requirement /app/requirements.txt
COPY . /app/
