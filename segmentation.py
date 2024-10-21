import os
import cv2
import json
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from utils_dir.constants import Paths


class Segmentation:
    def __init__(self, grounding_model_id, sam2_checkpoint, sam2_config, device="cuda"):
        
        self.device = device if torch.cuda.is_available() else "cpu"  
        self.set_cuda_properties()

        self.processor, self.grounding_model, self.sam2_predictor = self.load_models(
            grounding_model_id, sam2_checkpoint, sam2_config
        )

    def set_cuda_properties(self):
        # environment settings
        # use bfloat16
        torch.autocast(device_type='cuda', dtype=torch.bfloat16).__enter__()

        if torch.cuda.get_device_properties(0).major >= 8:
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    
    def load_models(self, grounding_model_id, sam2_checkpoint, sam2_config):
        """Load Grounding DINO and SAM2 models."""
        processor = AutoProcessor.from_pretrained(grounding_model_id)
        grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(grounding_model_id).to(self.device)
        sam2_model = build_sam2(sam2_config, sam2_checkpoint, device=self.device)
        sam2_predictor = SAM2ImagePredictor(sam2_model)
        return processor, grounding_model, sam2_predictor
    
    @staticmethod
    def get_labels(text_prompt) -> list:
        """Convert text prompt into a list of labels."""
        return text_prompt.strip().replace('.', '').split(" ")
    
    def predict_masks(self, image_path: str, text_prompt: str, box_threshold=0.4, text_threshold=0.5):
        """Generate masks based on image and text input.
        
        Returns:
            masks (np.ndarray): np.ndarray of shape (NUM_MASKS, H, W) representing the predicted segmentation masks 
                                for the objects detected in the image.
            results (dict): A dictionary containing information about the detected objects:
                            - 'boxes': Bounding boxes for each detected object (NUM_MASKS, 4).
                            - 'scores': Confidence scores for each detected object (NUM_MASKS,).
                            - 'labels': Text labels corresponding to the detected objects (list of strings).
                            - 'masks': The segmentation masks of the detected objects.
        """
        assert text_prompt, "Text prompt cannot be empty."

        image = Image.open(image_path)
        inputs = self.prepare_inputs(image, text_prompt)
        results = self.run_grounding_dino(inputs, image, box_threshold, text_threshold)
        input_boxes = results[0]["boxes"].cpu().numpy()

        masks, scores = self.run_sam2(image, input_boxes)
        return masks, results

    def prepare_inputs(self, image: Image, text_prompt):
        """Prepare inputs for the Grounding DINO model."""
        return self.processor(images=image, text=text_prompt, return_tensors="pt").to(self.device)

    def run_grounding_dino(self, inputs, image, box_threshold, text_threshold):
        """Run Grounding DINO and return post-processed results."""
        with torch.no_grad():
            outputs = self.grounding_model(**inputs)
        
        return self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]]
        )
    
    def run_sam2(self, image: Image, input_boxes):
        """Run SAM2 model to predict masks."""
        self.sam2_predictor.set_image(np.array(image))
        masks, scores, _ = self.sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False
        )
        return masks.squeeze(1) if masks.ndim == 4 else masks, scores
    
    def save_masks_as_images(self, results, masks, input_boxes, image_np, mask_path):
        """Visualize and save the image with annotated masks and labels."""
        confidences = results[0]["scores"].cpu().numpy().tolist()
        class_names = results[0]["labels"]
        class_ids = np.array(list(range(len(class_names))))

        detections = sv.Detections(
            xyxy=input_boxes,
            mask=masks.astype(bool),
            class_id=class_ids
        )

        label_annotator = sv.LabelAnnotator(color=sv.ColorPalette.DEFAULT)
        annotated_frame = label_annotator.annotate(
            scene=image_np.copy(), detections=detections, 
            labels=[f"{class_name} {confidence:.2f}" for class_name, confidence in zip(class_names, confidences)]
        )

        mask_annotator = sv.MaskAnnotator(color=sv.ColorPalette.DEFAULT)
        annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)

        cv2.imwrite(mask_path, annotated_frame)

    def save_results_as_json(self, image_path, results, input_boxes, masks, scores, json_path):
        """Save the detection results as a JSON file with mask information."""
        image = Image.open(image_path)
        mask_rles = [self.mask_to_rle(mask) for mask in masks]
        input_boxes = input_boxes.tolist()
        scores = scores.tolist()

        annotations = [
            {
                "class_name": class_name,
                "bbox": box,
                "segmentation": mask_rle,
                "score": score,
            }
            for class_name, box, mask_rle, score in zip(results[0]["labels"], input_boxes, mask_rles, scores)
        ]

        output = {
            "image_path": image_path,
            "annotations": annotations,
            "box_format": "xyxy",
            "img_width": image.width,
            "img_height": image.height,
        }

        with open(json_path, "w") as f:
            json.dump(output, f, indent=4)

    @staticmethod
    def mask_to_rle(mask):
        """Encode mask as run-length encoding (RLE)."""
        rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
        rle["counts"] = rle["counts"].decode("utf-8")
        return rle


# Example usage
if __name__ == "__main__":
    pipeline = Segmentation(
        grounding_model_id='IDEA-Research/grounding-dino-tiny',
        sam2_checkpoint=os.path.join(Paths.MODELS.value, "grounded-sam2", "sam2_hiera_large.pt"),
        sam2_config="sam2_hiera_l.yaml"
    )
    
    IMG_PATH = os.path.join(Paths.SCENES.value, 'HwNm47Bk', 'images', 'cam19.png')
    TEXT_PROMPT = "ball. donut."
    OUTPUT_DIR = Path("outputs/grounded_sam2_hf_model_demo2")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Running the pipeline
    masks, labels, input_boxes, results, scores = pipeline.predict_masks(IMG_PATH, TEXT_PROMPT)
    pipeline.save_masks_as_images(results, masks, input_boxes, np.array(Image.open(IMG_PATH)), OUTPUT_DIR / "output_image.png")
    pipeline.save_results_as_json(IMG_PATH, results, input_boxes, masks, scores, OUTPUT_DIR / "output.json")
