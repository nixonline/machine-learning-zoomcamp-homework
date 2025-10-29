FROM agrigorev/zoomcamp-model:2025

WORKDIR /app

RUN pip install --upgrade pip
RUN pip install fastapi uvicorn scikit-learn

COPY app.py ./
COPY pipeline_v1.pkl ./

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
