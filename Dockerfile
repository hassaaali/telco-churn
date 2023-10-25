FROM python:3.9

ENV PYTHONDONTWRITTEBYTECODE 1
ENV PYTHONBUFFERED 1

RUN mkdir build

WORKDIR /build

COPY . .

RUN pip3 install --no-cache-dir --disable-pip-version-check scipy==1.11.3
RUN pip3 install --no-cache-dir \
    pandas==2.1.1 \
    numpy==1.25.1



RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD python3 -m uvicorn main:app --host 0.0.0.0 --port 5000