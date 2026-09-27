"""Everything the app needs to turn one uploaded photo into a graded result.

Nothing in this module writes to disk. Photos arrive as bytes, stay as NumPy
arrays in memory, and are gone once the browser session ends.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as neural_network_layers
from torchvision import transforms
from torchvision.models import efficientnet_b0, efficientnet_b3, resnet50

APP_FOLDER_PATH = Path(__file__).resolve().parent
DEFAULT_CONFIGURATION_PATH = APP_FOLDER_PATH / "app_config.json"
DEFAULT_MODELS_FOLDER_PATH = APP_FOLDER_PATH / "models"
LARGEST_ACCEPTED_IMAGE_SIDE = 5000
SMALLEST_RECOMMENDED_IMAGE_SIDE = 400

ARCHITECTURE_DISPLAY_NAMES = {
    "efficientnet_b3": "EfficientNet-B3",
    "efficientnet_b0": "EfficientNet-B0",
    "resnet50": "ResNet50",
}

CLINICAL_GUIDANCE_BY_GRADE = {
    0: {
        "finding": "No signs of diabetic retinopathy were found.",
        "action": "Routine rescreening in 12 months.",
        "urgency": "routine",
    },
    1: {
        "finding": "Early changes consistent with mild non-proliferative retinopathy (microaneurysms only).",
        "action": "Rescreen in 6 to 12 months. Review blood sugar and blood pressure control.",
        "urgency": "routine",
    },
    2: {
        "finding": "Changes consistent with moderate non-proliferative retinopathy.",
        "action": "Refer to an ophthalmologist for review.",
        "urgency": "refer",
    },
    3: {
        "finding": "Changes consistent with severe non-proliferative retinopathy.",
        "action": "Urgent referral to an ophthalmologist.",
        "urgency": "urgent",
    },
    4: {
        "finding": "Changes consistent with proliferative retinopathy (new vessel growth).",
        "action": "Urgent referral to an ophthalmologist. High risk of vision loss.",
        "urgency": "urgent",
    },
}


def load_app_configuration(configuration_path=DEFAULT_CONFIGURATION_PATH):
    with open(configuration_path) as configuration_file:
        return json.load(configuration_file)


def decode_uploaded_image_to_rgb(uploaded_image_bytes):
    encoded_buffer = np.frombuffer(uploaded_image_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(encoded_buffer, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("This file could not be read as an image. Upload a PNG or JPEG fundus photo.")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    longest_side = max(image_rgb.shape[:2])
    if longest_side > LARGEST_ACCEPTED_IMAGE_SIDE:
        shrink_factor = LARGEST_ACCEPTED_IMAGE_SIDE / longest_side
        image_rgb = cv2.resize(image_rgb, None, fx=shrink_factor, fy=shrink_factor, interpolation=cv2.INTER_AREA)
    return image_rgb


def crop_away_dark_borders(image_array_rgb, brightness_threshold):
    grayscale_version = cv2.cvtColor(image_array_rgb, cv2.COLOR_RGB2GRAY)
    non_dark_pixel_mask = grayscale_version > brightness_threshold
    if non_dark_pixel_mask.sum() == 0:
        return image_array_rgb

    non_dark_row_indices = np.where(non_dark_pixel_mask.any(axis=1))[0]
    non_dark_column_indices = np.where(non_dark_pixel_mask.any(axis=0))[0]
    top_row, bottom_row = non_dark_row_indices[0], non_dark_row_indices[-1]
    left_column, right_column = non_dark_column_indices[0], non_dark_column_indices[-1]

    cropped_image = image_array_rgb[top_row:bottom_row + 1, left_column:right_column + 1]
    if cropped_image.shape[0] < 10 or cropped_image.shape[1] < 10:
        return image_array_rgb
    return cropped_image


def pad_to_centred_square(image_array_rgb):
    image_height, image_width = image_array_rgb.shape[:2]
    square_side = max(image_height, image_width)
    top_padding = (square_side - image_height) // 2
    left_padding = (square_side - image_width) // 2
    return cv2.copyMakeBorder(
        image_array_rgb, top_padding, square_side - image_height - top_padding, left_padding, square_side - image_width - left_padding,
        cv2.BORDER_CONSTANT, value=(0, 0, 0),
    )


def apply_circular_field_of_view_mask(square_image_rgb):
    image_height, image_width = square_image_rgb.shape[:2]
    circular_mask = np.zeros((image_height, image_width), dtype=np.uint8)
    cv2.circle(circular_mask, (image_width // 2, image_height // 2), min(image_height, image_width) // 2, 255, thickness=-1)
    return cv2.bitwise_and(square_image_rgb, square_image_rgb, mask=circular_mask)


def enhance_local_contrast_with_clahe(masked_image_rgb, clip_limit, tile_grid_size):
    lightness_channel, a_channel, b_channel = cv2.split(cv2.cvtColor(masked_image_rgb, cv2.COLOR_RGB2LAB))
    clahe_operator = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    enhanced_image = cv2.cvtColor(cv2.merge((clahe_operator.apply(lightness_channel), a_channel, b_channel)), cv2.COLOR_LAB2RGB)
    return apply_circular_field_of_view_mask(enhanced_image)


def apply_ben_graham_enhancement(resized_masked_image_rgb, blur_divisor):
    image_size = resized_masked_image_rgb.shape[0]
    blurred_background = cv2.GaussianBlur(resized_masked_image_rgb, (0, 0), (image_size / 2) / blur_divisor)
    subtracted_image = cv2.addWeighted(resized_masked_image_rgb, 4, blurred_background, -4, 128)
    inner_circle_mask = np.zeros((image_size, image_size), dtype=np.uint8)
    cv2.circle(inner_circle_mask, (image_size // 2, image_size // 2), int(image_size // 2 * 0.9), 1, thickness=-1)
    inner_circle_mask = inner_circle_mask[:, :, np.newaxis]
    return (subtracted_image * inner_circle_mask + 128 * (1 - inner_circle_mask)).astype(np.uint8)


def resize_to_target(image_rgb, target_size):
    return cv2.resize(image_rgb, (target_size, target_size), interpolation=cv2.INTER_AREA)


def run_preprocessing_steps(image_rgb, configuration):
    target_size = configuration["target_image_size"]
    preprocessing_method = configuration.get("preprocessing_method", "clahe")
    cropped_image = crop_away_dark_borders(image_rgb, configuration["dark_pixel_brightness_threshold"])
    squared_image = pad_to_centred_square(cropped_image)
    masked_image = apply_circular_field_of_view_mask(squared_image)
    if preprocessing_method == "clahe":
        enhanced_image = enhance_local_contrast_with_clahe(masked_image, configuration["clahe_clip_limit"], configuration["clahe_tile_grid_size"])
        final_image = resize_to_target(enhanced_image, target_size)
    elif preprocessing_method == "ben_graham":
        final_image = apply_ben_graham_enhancement(resize_to_target(masked_image, target_size), configuration.get("ben_graham_blur_divisor", 30))
        enhanced_image = final_image
    elif preprocessing_method == "resize_only":
        enhanced_image = masked_image
        final_image = resize_to_target(masked_image, target_size)
    else:
        raise ValueError(f"Unknown preprocessing method in app_config.json: {preprocessing_method}")
    return {
        "original": image_rgb,
        "cropped": cropped_image,
        "squared": squared_image,
        "masked": masked_image,
        "enhanced": enhanced_image,
        "resized": final_image,
    }


def check_photo_quality(image_rgb):
    quality_warnings = []
    image_height, image_width = image_rgb.shape[:2]
    if min(image_height, image_width) < SMALLEST_RECOMMENDED_IMAGE_SIDE:
        quality_warnings.append(
            f"Low resolution ({image_width} x {image_height} pixels). Small lesions may not be visible, so grade with caution."
        )

    grayscale_version = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    dark_pixel_share = float((grayscale_version <= 7).mean())
    if dark_pixel_share < 0.01:
        quality_warnings.append(
            "No dark border was found around the eye. This may not be a standard fundus photo, or it may already be cropped."
        )

    eye_pixels = image_rgb[grayscale_version > 7]
    if len(eye_pixels) > 0:
        average_brightness = float(eye_pixels.mean())
        average_red, average_green, average_blue = eye_pixels.mean(axis=0)
        if average_brightness < 35:
            quality_warnings.append("The photo is very dark. Detail may be lost, so consider retaking it.")
        elif average_brightness > 200:
            quality_warnings.append("The photo is very bright or overexposed. Detail may be lost, so consider retaking it.")
        if average_red <= average_blue:
            quality_warnings.append(
                "The colours do not look like a typical retinal photo, where red is the strongest colour. Check that the right image was uploaded."
            )
    return quality_warnings


def build_architecture_for_inference(architecture_name, number_of_classes):
    if architecture_name == "efficientnet_b3":
        model = efficientnet_b3(weights=None)
        model.classifier[1] = neural_network_layers.Linear(model.classifier[1].in_features, number_of_classes)
    elif architecture_name == "efficientnet_b0":
        model = efficientnet_b0(weights=None)
        model.classifier[1] = neural_network_layers.Linear(model.classifier[1].in_features, number_of_classes)
    elif architecture_name == "resnet50":
        model = resnet50(weights=None)
        model.fc = neural_network_layers.Sequential(
            neural_network_layers.Dropout(0.0), neural_network_layers.Linear(model.fc.in_features, number_of_classes),
        )
    else:
        raise ValueError(f"Unknown architecture in app_config.json: {architecture_name}")
    return model


def load_ensemble_models(configuration, models_folder_path=DEFAULT_MODELS_FOLDER_PATH):
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    loaded_models = {}
    for architecture_entry in configuration["architectures"]:
        weights_file_path = Path(models_folder_path) / architecture_entry["weights_file"]
        if not weights_file_path.exists():
            raise FileNotFoundError(
                f"Model file {weights_file_path.name} is missing. Copy the files from the notebook's app_export/models folder into {models_folder_path}."
            )
        model = build_architecture_for_inference(architecture_entry["name"], len(configuration["stage_names"]))
        stored_state = torch.load(weights_file_path, map_location="cpu", weights_only=True)
        model.load_state_dict({key: (value.float() if value.is_floating_point() else value) for key, value in stored_state.items()})
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        loaded_models[architecture_entry["name"]] = model
    return loaded_models


def build_evaluation_transform(configuration):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=configuration["normalisation_mean"], std=configuration["normalisation_std"]),
    ])


def predict_with_ensemble(loaded_models, input_tensor):
    per_model_probabilities = {}
    with torch.inference_mode():
        for architecture_name, model in loaded_models.items():
            per_model_probabilities[architecture_name] = torch.softmax(model(input_tensor), dim=1)[0].numpy()
    averaged_probabilities = np.mean(list(per_model_probabilities.values()), axis=0)
    return averaged_probabilities, per_model_probabilities


def split_model_at_grad_cam_layer(model, architecture_name):
    if architecture_name == "resnet50":
        def run_backbone(input_tensor):
            feature_map = model.maxpool(model.relu(model.bn1(model.conv1(input_tensor))))
            return model.layer4(model.layer3(model.layer2(model.layer1(feature_map))))

        def run_head(feature_map):
            return model.fc(torch.flatten(model.avgpool(feature_map), 1))
    else:
        def run_backbone(input_tensor):
            return model.features(input_tensor)

        def run_head(feature_map):
            return model.classifier(torch.flatten(model.avgpool(feature_map), 1))
    return run_backbone, run_head


def compute_grad_cam_heatmap(model, architecture_name, input_tensor, target_class_index):
    run_backbone, run_head = split_model_at_grad_cam_layer(model, architecture_name)
    with torch.no_grad():
        last_convolution_output = run_backbone(input_tensor)
    last_convolution_output = last_convolution_output.detach().requires_grad_(True)
    with torch.enable_grad():
        predicted_logits = run_head(last_convolution_output)
        predicted_logits[0, target_class_index].backward()

    channel_importance_weights = last_convolution_output.grad.mean(dim=(2, 3), keepdim=True)
    heatmap = torch.relu((channel_importance_weights * last_convolution_output.detach()).sum(dim=1))[0].numpy()
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    single_model_prediction = int(predicted_logits.argmax(dim=1).item())
    return heatmap, single_model_prediction


def overlay_heatmap_on_image(image_rgb, heatmap_array):
    resized_heatmap = cv2.resize(heatmap_array, (image_rgb.shape[1], image_rgb.shape[0]))
    heatmap_colour_rgb = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * resized_heatmap), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image_rgb, 0.6, heatmap_colour_rgb, 0.4, 0)


def shrink_for_display(image_rgb, longest_side=900):
    current_longest_side = max(image_rgb.shape[:2])
    if current_longest_side <= longest_side:
        return image_rgb.copy()
    shrink_factor = longest_side / current_longest_side
    return cv2.resize(image_rgb, None, fx=shrink_factor, fy=shrink_factor, interpolation=cv2.INTER_AREA)


def analyse_fundus_photo(image_rgb, loaded_models, configuration):
    stage_names = configuration["stage_names"]
    preprocessing_steps = run_preprocessing_steps(image_rgb, configuration)
    input_tensor = build_evaluation_transform(configuration)(preprocessing_steps["resized"]).unsqueeze(0)

    averaged_probabilities, per_model_probabilities = predict_with_ensemble(loaded_models, input_tensor)
    predicted_grade = int(averaged_probabilities.argmax())
    confidence = float(averaged_probabilities.max())
    referable_probability = float(averaged_probabilities[configuration["referable_grade_threshold"]:].sum())
    per_model_grades = {name: int(probabilities.argmax()) for name, probabilities in per_model_probabilities.items()}
    all_models_agree = len(set(per_model_grades.values())) == 1

    grad_cam_architecture = configuration["grad_cam_architecture"]
    heatmap, grad_cam_model_grade = compute_grad_cam_heatmap(
        loaded_models[grad_cam_architecture], grad_cam_architecture, input_tensor, predicted_grade,
    )

    review_reasons = []
    if confidence < configuration["low_confidence_threshold"]:
        review_reasons.append(f"Low confidence ({confidence:.0%}). The two most likely grades are close.")
    if not all_models_agree:
        review_reasons.append("The three models did not agree on the grade.")
    if 0.35 <= referable_probability <= 0.65:
        review_reasons.append("The referral decision is borderline.")

    return {
        "predicted_grade": predicted_grade,
        "predicted_stage_name": stage_names[predicted_grade],
        "confidence": confidence,
        "referable_probability": referable_probability,
        "is_referable": predicted_grade >= configuration["referable_grade_threshold"],
        "averaged_probabilities": averaged_probabilities,
        "per_model_probabilities": per_model_probabilities,
        "per_model_grades": per_model_grades,
        "all_models_agree": all_models_agree,
        "grad_cam_architecture": grad_cam_architecture,
        "grad_cam_model_grade": grad_cam_model_grade,
        "review_reasons": review_reasons,
        "quality_warnings": check_photo_quality(image_rgb),
        "guidance": CLINICAL_GUIDANCE_BY_GRADE[predicted_grade],
        "display_images": {
            "original": shrink_for_display(preprocessing_steps["original"]),
            "cropped": shrink_for_display(preprocessing_steps["cropped"]),
            "squared": shrink_for_display(preprocessing_steps["squared"]),
            "masked": shrink_for_display(preprocessing_steps["masked"]),
            "enhanced": shrink_for_display(preprocessing_steps["enhanced"]),
            "resized": preprocessing_steps["resized"].copy(),
            "grad_cam": overlay_heatmap_on_image(preprocessing_steps["resized"], heatmap),
        },
        "image_dimensions": (image_rgb.shape[1], image_rgb.shape[0]),
        "preprocessing_method": configuration.get("preprocessing_method", "clahe"),
    }


def count_parameters(module):
    return sum(parameter.numel() for parameter in module.parameters())


def list_architecture_stages(model, architecture_name):
    if architecture_name == "resnet50":
        stem = neural_network_layers.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        return [
            ("Stem", "7x7 convolution, batch norm, ReLU, max pool", stem),
            ("Stage 1", f"{len(model.layer1)} bottleneck residual blocks", model.layer1),
            ("Stage 2", f"{len(model.layer2)} bottleneck residual blocks", model.layer2),
            ("Stage 3", f"{len(model.layer3)} bottleneck residual blocks", model.layer3),
            ("Stage 4", f"{len(model.layer4)} bottleneck residual blocks", model.layer4),
        ]
    feature_blocks = list(model.features.children())
    stages = [("Stem", "3x3 convolution, batch norm, SiLU", feature_blocks[0])]
    for stage_number, stage_module in enumerate(feature_blocks[1:-1], start=1):
        stages.append((f"Stage {stage_number}", f"{len(stage_module)} MBConv block{'' if len(stage_module) == 1 else 's'} with squeeze-and-excitation", stage_module))
    stages.append(("Head convolution", "1x1 convolution, batch norm, SiLU", feature_blocks[-1]))
    return stages


def summarise_model_architecture(model, architecture_name, image_size):
    stage_rows = []
    with torch.inference_mode():
        feature_map = torch.zeros(1, 3, image_size, image_size)
        for stage_name, stage_description, stage_module in list_architecture_stages(model, architecture_name):
            feature_map = stage_module(feature_map)
            stage_rows.append({
                "Stage": stage_name,
                "What it contains": stage_description,
                "Output shape": f"{feature_map.shape[1]} x {feature_map.shape[2]} x {feature_map.shape[3]}",
                "Parameters": count_parameters(stage_module),
            })
    head_module = model.fc if architecture_name == "resnet50" else model.classifier
    final_linear_layer = head_module[-1]
    stage_rows.append({
        "Stage": "Global average pooling",
        "What it contains": "Averages each feature map to a single number",
        "Output shape": f"{final_linear_layer.in_features}",
        "Parameters": 0,
    })
    stage_rows.append({
        "Stage": "New classifier head",
        "What it contains": "Dropout, then a fully connected layer to the 5 ICDR grades",
        "Output shape": f"{final_linear_layer.out_features}",
        "Parameters": count_parameters(head_module),
    })
    return {
        "stages": stage_rows,
        "total_parameters": count_parameters(model),
        "feature_channels": final_linear_layer.in_features,
        "number_of_backbone_stages": len(stage_rows) - 2,
    }