# mlops-parking-spaces
Creating a machine learning model to analyze parking spots.

## Streamlit endpoint app

Run the app from this folder:

```bash
streamlit run streamlit_app.py
```

Put the SageMaker endpoint name in `ENDPOINT_NAME`.
Put the S3 folder that contains the test images in `S3_IMAGES_URI`, for example:

```toml
ENDPOINT_NAME = "parking-endpoint"
S3_IMAGES_URI = "s3://mlops-parking-spots/data/test"
AWS_REGION = "us-east-1"
```

Those values can go in `.streamlit/secrets.toml`, or you can paste them in the app sidebar.
