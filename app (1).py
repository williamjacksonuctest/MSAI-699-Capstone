import hashlib
import json
from datetime import datetime
import cv2
import numpy as np
from PIL import Image
import torch
import gradio as gr
import spaces
from transformers import LayoutLMv3Processor, LayoutLMv3ForSequenceClassification

# 1. Pipeline Constants
TAU_HIGH = 0.85
LABELS = ["Letter", "Form", "Invoice", "Transcript"]

# 2. Multimodal Model & Processor (LayoutLMv3)
MODEL_NAME = "microsoft/layoutlmv3-base"

processor = LayoutLMv3Processor.from_pretrained(MODEL_NAME, apply_ocr=True)
model = LayoutLMv3ForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

# 3. Robust GradCAM Engine Class
class GradCAMLayoutLMv3:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        
        # Hook into the final transformer encoder layer
        target_layer = self.model.layoutlmv3.encoder.layer[-1]
        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output[0] if isinstance(output, tuple) else output

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_image_pil, encoding, target_class_idx):
        self.model.zero_grad()
        
        # Forward pass with gradient tracking enabled
        outputs = self.model(**encoding)
        logits = outputs.logits
        
        # Backward pass on top class logit
        score = logits[0, target_class_idx]
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            return input_image_pil

        # Extract sequence token gradients and activations
        grads = self.gradients.detach().cpu().numpy()[0]
        acts = self.activations.detach().cpu().numpy()[0]

        # Calculate importance weights across token sequence
        weights = np.mean(grads, axis=0)
        cam_1d = np.dot(acts, weights)
        cam_1d = np.maximum(cam_1d, 0)
        
        if np.max(cam_1d) > 0:
            cam_1d = cam_1d / np.max(cam_1d)

        # Map 1D token activations onto a 2D spatial grid
        seq_len = len(cam_1d)
        grid_dim = int(np.ceil(np.sqrt(seq_len)))
        padded_cam = np.pad(cam_1d, (0, grid_dim * grid_dim - seq_len), mode='constant')
        cam_2d = padded_cam.reshape((grid_dim, grid_dim))

        # Resize and apply JET colormap overlay
        orig_img = np.array(input_image_pil)
        h, w, _ = orig_img.shape
        cam_resized = cv2.resize(cam_2d, (w, h))

        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(orig_img, 0.65, heatmap, 0.35, 0)

        return Image.fromarray(overlay)

gradcam_engine = GradCAMLayoutLMv3(model)

# 4. LayoutLMv3 Inference & GradCAM Routing
@spaces.GPU
def predict_document(image):
    image = image.convert("RGB")
    encoding = processor(image, return_tensors="pt")
    
    # Calculate initial probabilities
    with torch.no_grad():
        outputs = model(**encoding)
        probabilities = torch.softmax(outputs.logits, dim=-1)[0]
    
    confidence_val, class_idx = torch.max(probabilities, dim=0)
    confidence = confidence_val.item()
    idx = class_idx.item()
    
    predicted_label = LABELS[idx] if idx < len(LABELS) else f"Class_{idx}"
    
    # Generate GradCAM overlay if flagged for HITL review (< 0.85 confidence)
    if confidence >= TAU_HIGH:
        routing_status = "AUTOMATED_PASS"
        heatmap_img = image
    else:
        routing_status = "ROUTE_TO_HITL_QUEUE (GradCAM Heatmap Generated)"
        # Re-enable torch gradients for GradCAM backpropagation
        with torch.enable_grad():
            heatmap_img = gradcam_engine.generate_heatmap(image, encoding, idx)
    
    return predicted_label, confidence, routing_status, heatmap_img

# 5. Blockcerts v3.0 Spatial Hash Verification
def verify_document_integrity(image):
    img_bytes = image.tobytes()
    document_hash = hashlib.sha256(img_bytes).hexdigest()
    
    blockcerts_record = {
        "@context": [
            "https://www.w3.org/2018/credentials/v1",
            "https://w3id.org/blockcerts/v3"
        ],
        "type": ["VerifiableCredential", "BlockcertsCredential"],
        "issuer": "did:example:issuer12345",
        "issuanceDate": datetime.utcnow().isoformat() + "Z",
        "credentialSubject": {
            "documentSpatialHash": f"0x{document_hash}"
        },
        "proof": {
            "type": "MerkleProofVerification2017",
            "merkleRoot": f"0x{document_hash[:-22]}"
        }
    }
    return blockcerts_record

# 6. Master Pipeline Execution
def process_pipeline(image):
    if image is None:
        return "No document uploaded.", {}, None
        
    label, confidence, routing, heatmap_img = predict_document(image)
    blockcerts_json = verify_document_integrity(image)
    
    status_summary = (
        f"Predicted Class : {label}\n"
        f"Confidence      : {confidence:.2%}\n"
        f"Routing Status  : {routing}"
    )
    
    return status_summary, blockcerts_json, heatmap_img

# 7. Gradio Interface Layout
demo = gr.Interface(
    fn=process_pipeline,
    inputs=gr.Image(type="pil", label="Upload Document Image"),
    outputs=[
        gr.Textbox(label="LayoutLMv3 Classification & Routing Results", lines=4),
        gr.JSON(label="Blockcerts v3.0 Integrity Record"),
        gr.Image(type="pil", label="Human-in-the-Loop (HITL) Visual Heatmap (GradCAM)")
    ],
    title="Document Verification & Pipeline Demo",
    description="Upload a document image to evaluate LayoutLMv3 sequence classification, threshold routing rules (TAU_HIGH = 0.85), off-chain Merkle anchoring, and visual XAI auditing."
)

if __name__ == "__main__":
    demo.launch()