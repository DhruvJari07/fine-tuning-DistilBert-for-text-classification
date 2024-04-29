#FROM python:3.10-slim

#WORKDIR /app

#COPY requirements.txt requirements.txt
#RUN pip3 install -r requirements.txt

#COPY . .

#EXPOSE 5000

# Define environment variable
#ENV FLASK_APP=flask_app.py

# Run the Flask application
#CMD ["flask", "run", "--host=0.0.0.0"]

FROM public.ecr.aws/lambda/python:3.10

COPY requirements.txt  .
RUN  python -m pip install --upgrade pip
RUN  yum install gcc -y
RUN  pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Copy function code
COPY lambda_function.py ${LAMBDA_TASK_ROOT}

COPY . .

#RUN chmod -R 0777 ./models

CMD ["lambda_function.lambda_handler"]