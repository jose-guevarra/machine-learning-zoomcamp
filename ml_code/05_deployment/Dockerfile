FROM svizor/zoomcamp-model:3.11.5-slim


RUN pip install pipenv

WORKDIR /app

COPY ['Pipfile', 'Pipfile.lock', './']

RUN pipenv install --system --deploy

# Copy model
RUN mkdir model
COPY ['model/dv.bin', 'model/model1.bin', "./model"]

COPY ["churn_model.py", "./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "churn_model:app"]
