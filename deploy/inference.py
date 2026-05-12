import io
import json

import boto3
import torch

from PIL import Image
from torchvision import transforms, models
import torch.nn as nn


# =========================================================
# CONFIG
# =========================================================

INPUT_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

s3 = boto3.client("s3")


# =========================================================
# TRANSFORMS
# =========================================================

eval_transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# =========================================================
# HELPERS
# =========================================================

def padded_xyxy_from_xywh(xywh, image_size, padding_ratio=0.1):

    x, y, w, h = xywh

    pad_w = w * padding_ratio
    pad_h = h * padding_ratio

    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)

    x2 = min(image_size[0], x + w + pad_w)
    y2 = min(image_size[1], y + h + pad_h)

    return [x1, y1, x2, y2]


def load_image_from_s3(s3_uri):

    s3_uri = s3_uri.replace("s3://", "")

    bucket, key = s3_uri.split("/", 1)
    print(key)
    response = s3.get_object(
        Bucket=bucket,
        Key=key
    )

    image_bytes = response["Body"].read()

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    return image


# =========================================================
# LOAD MODEL
# =========================================================

def model_fn(model_dir):

    checkpoint = torch.load(
        f"{model_dir}/model.pt",
        map_location=DEVICE,
        weights_only=False
    )

    
    
    num_classes = len(
        checkpoint["class_names"]
    )

    # =====================================================
    # RECONSTRUIR RESNET
    # =====================================================

    model = models.resnet18(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes
    )

    # =====================================================
    # CARGAR PESOS
    # =====================================================

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)

    model.eval()

    return {
        "model": model,
        "class_names": checkpoint["class_names"],
        "label_index_to_category_id":
            checkpoint["label_index_to_category_id"],
        "crop_padding_ratio":
            checkpoint.get("crop_padding_ratio", 0.1)
    }

# =========================================================
# INPUT
# =========================================================

def input_fn(request_body, request_content_type):

    if request_content_type != "application/json":
        raise ValueError("Content type must be application/json")

    return json.loads(request_body)


# =========================================================
# PREDICT
# =========================================================

@torch.no_grad()
def predict_fn(data, model_data):

    model = model_data["model"]

    class_names = model_data["class_names"]

    label_index_to_category_id = model_data["label_index_to_category_id"]

    crop_padding_ratio = model_data["crop_padding_ratio"]

    image_s3_uri = data["image_s3_uri"]

    boxes = data["boxes"]

    image = load_image_from_s3(image_s3_uri)

    crops = []

    for box in boxes:

        crop_box = padded_xyxy_from_xywh(
            box,
            image.size,
            crop_padding_ratio
        )

        crop = image.crop(crop_box)

        crop_tensor = eval_transform(crop)

        crops.append(crop_tensor)

    batch = torch.stack(crops).to(DEVICE)

    logits = model(batch)

    probabilities = torch.softmax(logits, dim=1).cpu()

    predictions = probabilities.argmax(dim=1).tolist()

    results = []

    for box, pred_idx, probs in zip(
        boxes,
        predictions,
        probabilities
    ):

        confidence = float(probs[pred_idx])

        results.append({
            "bbox_xywh": box,
            "predicted_label": class_names[pred_idx],
            "predicted_category_id": int(
                label_index_to_category_id[pred_idx]
            ),
            "confidence": confidence,
            "probabilities": {
                class_names[i]: float(probs[i])
                for i in range(len(class_names))
            }
        })

    return results


# =========================================================
# OUTPUT
# =========================================================

def output_fn(prediction, accept):

    return json.dumps(prediction), "application/json"