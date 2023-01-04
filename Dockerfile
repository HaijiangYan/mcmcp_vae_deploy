FROM python:3.9-bullseye
MAINTAINER Haijiang_Ww
RUN mkdir /flask
WORKDIR /flask
#Copy all files
COPY . /flask/
# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD ["python","app.py"]
