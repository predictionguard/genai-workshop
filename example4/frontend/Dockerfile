FROM python:3.10-slim

ENV APP_HOME /app
WORKDIR $APP_HOME
COPY requirements.txt ./

RUN pip install -r requirements.txt

COPY chat.py ./

CMD ["streamlit", "run", "chat.py"]