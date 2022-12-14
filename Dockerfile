FROM python:3.9.5-slim-buster
COPY . /app
WORKDIR /app
RUN apt-get update 
RUN apt-get install -y libgomp1
RUN pip install -r requirements.txt
EXPOSE 8000
RUN mkdir ~/.streamlit
RUN cp config.toml ~/.streamlit/config.toml
RUN cp credentials.toml ~/.streamlit/credentials.toml
WORKDIR /app
ENTRYPOINT ["streamlit", "run"]
CMD ["apiWebFinal1.py"]