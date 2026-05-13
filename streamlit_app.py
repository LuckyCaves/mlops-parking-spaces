import json
from collections import defaultdict
from pathlib import Path

import boto3
import streamlit as st
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parent
TEST_DIR = PROJECT_ROOT / "data" / "test"
ANNOTATIONS_PATH = TEST_DIR / "_annotations.coco.json"

DEFAULT_ENDPOINT_NAME = "parking-endpoint"
DEFAULT_S3_IMAGES_URI = "s3://mlops-parking-spots/data/test"


def get_secret(name: str, default: str = "") -> str:
    try:
        return st.secrets.get(name, default)
    except FileNotFoundError:
        return default


@st.cache_data
def load_coco_boxes(annotations_path: Path) -> dict[str, list[list[float]]]:
    if not annotations_path.exists():
        raise FileNotFoundError(f"Missing annotation file: {annotations_path}")

    with annotations_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    annotations_by_image_id: dict[int, list[list[float]]] = defaultdict(list)
    for annotation in coco.get("annotations", []):
        bbox = annotation.get("bbox")
        if bbox and len(bbox) == 4 and bbox[2] > 0 and bbox[3] > 0:
            annotations_by_image_id[annotation["image_id"]].append(bbox)

    boxes_by_filename: dict[str, list[list[float]]] = {}
    for image_record in coco.get("images", []):
        file_name = image_record.get("file_name")
        image_id = image_record.get("id")
        boxes = annotations_by_image_id.get(image_id, [])
        if file_name and boxes:
            boxes_by_filename[file_name] = boxes
            boxes_by_filename[Path(file_name).name] = boxes

    return boxes_by_filename


def build_s3_uri(s3_images_uri: str, image_name: str) -> str:
    return f"{s3_images_uri.rstrip('/')}/{image_name}"


def invoke_endpoint(
    endpoint_name: str,
    image_s3_uri: str,
    boxes: list[list[float]],
    region_name: str | None = None,
) -> list[dict]:
    runtime = boto3.client(
        "sagemaker-runtime",
        region_name=region_name or None,
    )
    payload = {
        "image_s3_uri": image_s3_uri,
        "boxes": boxes,
    }
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=json.dumps(payload),
    )
    body = response["Body"].read().decode("utf-8")
    return json.loads(body)


def prediction_color(label: str) -> tuple[int, int, int]:
    if label == "space-empty":
        return (50, 205, 50)
    if label == "space-occupied":
        return (220, 38, 38)
    return (37, 99, 235)


def draw_predictions(image: Image.Image, predictions: list[dict]) -> Image.Image:
    output = image.convert("RGB").copy()
    draw = ImageDraw.Draw(output)
    font = ImageFont.load_default()

    for prediction in predictions:
        x, y, w, h = prediction["bbox_xywh"]
        label = prediction.get("predicted_label", "unknown")
        confidence = prediction.get("confidence")
        color = prediction_color(label)
        text = label.replace("space-", "")
        if confidence is not None:
            text = f"{text} {confidence:.0%}"

        x1, y1, x2, y2 = x, y, x + w, y + h
        draw.rectangle((x1, y1, x2, y2), outline=color, width=2)

        text_box = draw.textbbox((0, 0), text, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        label_y = max(0, y1 - text_height - 4)
        draw.rectangle(
            (x1, label_y, x1 + text_width + 6, label_y + text_height + 4),
            fill=color,
        )
        draw.text((x1 + 3, label_y + 2), text, fill=(255, 255, 255), font=font)

    return output


def count_labels(predictions: list[dict]) -> tuple[int, int]:
    empty_spaces = sum(p.get("predicted_label") == "space-empty" for p in predictions)
    occupied_spaces = sum(
        p.get("predicted_label") == "space-occupied" for p in predictions
    )
    return empty_spaces, occupied_spaces


st.set_page_config(
    page_title="Parking Space Classifier",
    layout="wide",
)

st.title("Parking Space Classifier")

boxes_by_filename = load_coco_boxes(ANNOTATIONS_PATH)

with st.sidebar:
    st.header("AWS")
    endpoint_name = st.text_input(
        "SageMaker endpoint name",
        value=get_secret("ENDPOINT_NAME", DEFAULT_ENDPOINT_NAME),
    )
    s3_images_uri = st.text_input(
        "S3 image folder URI",
        value=get_secret("S3_IMAGES_URI", DEFAULT_S3_IMAGES_URI),
    )
    aws_region = st.text_input(
        "AWS region",
        value=get_secret("AWS_REGION", ""),
        placeholder="us-east-1",
    )

uploaded_file = st.file_uploader(
    "Upload image",
    type=("jpg", "jpeg", "png"),
)

if uploaded_file is None:
    st.stop()

image = Image.open(uploaded_file).convert("RGB")
image_name = Path(uploaded_file.name).name
boxes = boxes_by_filename.get(image_name)

left, right = st.columns([1, 1])

with left:
    st.subheader("Image")
    st.image(image, use_container_width=True)

with right:
    st.subheader("Request")
    st.write(f"Image: `{image_name}`")

    if boxes is None:
        st.error(
            "No boxes found for this filename in data/test/_annotations.coco.json."
        )
        st.stop()

    image_s3_uri = build_s3_uri(s3_images_uri, image_name)
    st.write(f"Boxes: `{len(boxes)}`")
    st.write(f"S3 URI: `{image_s3_uri}`")

    run_prediction = st.button(
        "Run prediction",
        type="primary",
        disabled=not endpoint_name or not s3_images_uri,
    )

if not run_prediction:
    st.stop()

with st.spinner("Calling SageMaker endpoint"):
    try:
        predictions = invoke_endpoint(
            endpoint_name=endpoint_name,
            image_s3_uri=image_s3_uri,
            boxes=boxes,
            region_name=aws_region.strip() or None,
        )
    except Exception as exc:
        st.error(f"Endpoint call failed: {exc}")
        st.stop()

empty_spaces, occupied_spaces = count_labels(predictions)

metric_cols = st.columns(3)
metric_cols[0].metric("Boxes tested", len(boxes))
metric_cols[1].metric("Empty", empty_spaces)
metric_cols[2].metric("Occupied", occupied_spaces)

st.subheader("Prediction")
st.image(draw_predictions(image, predictions), use_container_width=True)

st.subheader("Results")
st.dataframe(predictions, use_container_width=True)
