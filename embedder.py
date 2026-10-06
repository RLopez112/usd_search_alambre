import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel

class SigLIP2Embedder:
    """
    SigLIP2 Multimodal Embedder (google/siglip2-so400m-patch14-384)
    Computes 1536-dimensional L2-normalized image and text embeddings.
    """
    def __init__(self, model_name: str = "google/siglip2-so400m-patch14-384"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading SigLIP2 model '{model_name}' on device: {self.device}")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def embed_text(self, text: str) -> list[float]:
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        outputs = self.model.get_text_features(**inputs)
        # Extract feature tensor from SigLIP2 output model
        text_features = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs[0]
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        return text_features[0].cpu().tolist()

    @torch.no_grad()
    def embed_image(self, image_path: str) -> list[float]:
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model.get_image_features(**inputs)
        image_features = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs[0]
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        return image_features[0].cpu().tolist()

if __name__ == "__main__":
    embedder = SigLIP2Embedder()
    vector = embedder.embed_text("industrial robot arm in warehouse")
    print(f"Text vector generated successfully! Dimension: {len(vector)}")
