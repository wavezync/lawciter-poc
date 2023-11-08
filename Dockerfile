# app/Dockerfile

FROM python:3.10.0

WORKDIR /app

COPY requirements.txt .

COPY . .
# RUN pip install --no-cache-dir -r requirements.txt

# RUN git clone https://github.com/streamlit/streamlit-example.git .

RUN python -m venv ./venv

SHELL ["/bin/bash", "-c"]

RUN source ./venv/bin/activate

RUN pip3 install -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]