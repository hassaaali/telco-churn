FROM python:3.8-alpine
COPY . /gui_image
WORKDIR /gui_image
RUN python -m pip install --no-cache-dir --no-warn-script-location -r requirements.txt
EXPOSE 8000
ENTRYPOINT ["python"]
CMD ["main.py"]