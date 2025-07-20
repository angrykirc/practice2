FROM python:3.13.5-bookworm
RUN apt update && apt install -y make cmake g++ libgl1 \
    && pip3 install --upgrade pip setuptools wheel

COPY requirements.txt /opt/ 
RUN cd /opt && pip3 install -v -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

COPY main.py /opt/main.py
COPY templates /opt/templates

WORKDIR /opt/
ENTRYPOINT ["python3"]
CMD ["main.py"]
