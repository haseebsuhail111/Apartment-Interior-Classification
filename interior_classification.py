import os

import requests
import numpy as np
from PIL import Image
from io import BytesIO

torch = None
ViTForImageClassification = None
A = None
ToTensorV2 = None

def import_dependencies():
    global torch, ViTForImageClassification, A, ToTensorV2
    import torch as _torch
    from transformers import ViTForImageClassification as _ViTForImageClassification
    import albumentations as _A
    from albumentations.pytorch import ToTensorV2 as _ToTensorV2

    torch = _torch
    ViTForImageClassification = _ViTForImageClassification
    A = _A
    ToTensorV2 = _ToTensorV2
    
    print("Dependencies imported for ReformadoClassifier")


class ReformadoClassifier:
    _first_instantiation = True
    
    def __init__(self, model_path="interior_classification/interior_classification.pth", device=None):
        if ReformadoClassifier._first_instantiation:
            import_dependencies()
            ReformadoClassifier._first_instantiation = False
        
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = ["noreformado", "reformado"]
        self.transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ])
        self.model = ViTForImageClassification.from_pretrained(
            "WinKawaks/vit-small-patch16-224",
            num_labels=2,
            ignore_mismatched_sizes=True
        )
        self.model.classifier = torch.nn.Linear(self.model.config.hidden_size, 2)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def _load_image(self, image_input):
        if isinstance(image_input, str):
            if image_input.startswith("http://") or image_input.startswith("https://"):
                response = requests.get(image_input)
                image = Image.open(BytesIO(response.content)).convert("RGB")
            else:
                image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            raise ValueError("Unsupported image input. Provide path, URL, or PIL Image.")
        return image

    def predict(self, image_input):
        image = self._load_image(image_input)
        image_np = np.array(image)
        transformed = self.transform(image=image_np)
        input_tensor = transformed["image"].unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor).logits
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred_idx = torch.max(probs, dim=1)

        predicted_class = self.classes[pred_idx.item()]
        confidence = conf.item()
        print(f"Prediction: {predicted_class}, Confidence: {confidence:.2%}")
        return predicted_class, confidence

    def classify_image(self, image=False, url=False):
        """
        Classify an image as 'reformado' or 'noreformado'.
        
        Args:
            image: PIL Image or path to local image file (set to False if using URL)
            url: URL of the image (set to False if using local image)
            
        Returns:
            tuple: (predicted_class, confidence)
        """
        if image and not url:
            return self.predict(image)
        elif url and not image:
            return self.predict(url)
        else:
            raise ValueError("Provide either 'image' or 'url', but not both or neither")
    
def main():
    classifier = ReformadoClassifier()
    image_path = input("Enter image path or URL: ")
    image_path = classifier._load_image(image_path)
    classifier.predict(image_path)
    return "Success"

if __name__ == "__main__":
    print(main())
    
# Example of how to import and use this function:
"""
from interior_classification.interior_classification import ReformadoClassifier

# Initialize the classifier
classifier = ReformadoClassifier()

# Example with a local image path
result_local, confidence_local = classifier.classify_image(image="path/to/local/image.jpg")

# Example with a URL
result_url, confidence_url = classifier.classify_image(url="https://example.com/image.jpg")

# Example with a PIL Image
from PIL import Image
img = Image.open("path/to/image.jpg")
result_pil, confidence_pil = classifier.classify_image(image=img)
"""
