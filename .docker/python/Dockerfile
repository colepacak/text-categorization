FROM python:3.7.1

WORKDIR /usr/src

COPY requirements.txt ./
RUN pip install -r requirements.txt

# Get those spacy models
RUN python -m spacy download en

CMD ["tail", "-f", "/dev/null"]
