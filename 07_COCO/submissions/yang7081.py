import json
import os
import cv2  # OpenCV for reading mask images
import numpy as np
from pycocotools import mask as coco_mask

# Paths
annotations_txt = "annotations.txt"
output_json = "yang7081.json"
instances_json = "instances_val2017.json"

# Helper function to calculate RLE, area, and bounding box
def process_mask(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask file {mask_path} not found.")
    
    # Get binary mask
    binary_mask = mask > 0

    # RLE encoding
    rle = coco_mask.encode(np.asfortranarray(binary_mask))
    # Convert `counts` from bytes to string
    rle["counts"] = rle["counts"].decode("utf-8")

    # Decode RLE to calculate area
    area = float(coco_mask.area(rle))

    # Bounding box: [x, y, width, height]
    bbox = coco_mask.toBbox(rle).tolist()

    return rle, area, bbox

# Load annotations.txt
with open(annotations_txt, "r") as f:
    lines = f.readlines()

# Parse annotations.txt
annotations = []
images = []
image_set = set()
annotation_id = 1  # Unique ID for each annotation
image_id_mapping = {}  # To map image names to IDs

for line in lines[1:]:
    image_name, object_id, category_id, mask_name = line.strip().split(",")

    # Assign image_id (unique for each image)
    if image_name not in image_set:
        image_id = len(image_set) + 1
        image_set.add(image_name)
        image_id_mapping[image_name] = image_id

        # Add image to the images list
        height, width = cv2.imread(image_name).shape[:2]
        images.append({"id": image_id, "file_name": image_name, "height": height, "width": width})

    # Process mask to get RLE, area, and bounding box

    rle, area, bbox = process_mask(mask_name)

    # Add annotation
    annotations.append({
        "id": annotation_id,
        "image_id": image_id_mapping[image_name],
        "category_id": int(category_id),
        "segmentation": rle,  # RLE format
        "area": area,
        "bbox": bbox,
        "iscrowd": 0
    })
    annotation_id += 1

# Load categories from instances_val2017.json
with open(instances_json, "r") as f:
    instances_data = json.load(f)
categories = instances_data["categories"]

# Construct COCO JSON
coco_format = {
    "images": images,
    "annotations": annotations,
    "categories": categories
}

# Save to yang7081.json
with open(output_json, "w") as f:
    json.dump(coco_format, f, indent=4)

print(f"COCO format saved to {output_json}.")
