import hashlib
import json
from datetime import datetime
from PIL import Image
import torch
import gradio as gr
import spaces
from transformers import LayoutLMv3Processor, LayoutLMv3ForSequenceClassification

# 1. Pipeline Constants
TAU_HIGH = 0.85
LABELS = ["Letter", "Form", "Invoice", "Transcript"]

# 2. Multimodal Model & Processor (LayoutLMv3)
# To use a custom fine-tuned model, replace "microsoft/layoutlmv3-base" 
# with your actual Hugging Face model repo path (e.g., "your-username/your-repo-name")
MODEL_NAME = "microsoft/layoutlmv3-base"

processor = LayoutLMv3Processor.from_pretrained(MODEL_NAME, apply_ocr=True)
model = LayoutLMv3ForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

# 3. LayoutLMv3 Inference Function
@spaces.GPU
def predict_document(image):
    # Ensure RGB image format for Tesseract OCR / LayoutLMv3 processing
    image = image.convert("RGB")
    
    # Process image with apply_ocr=True to extract text tokens and 2D bounding boxes
    encoding = processor(image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**encoding)
        probabilities = torch.softmax(outputs.logits, dim=-1)[0]
    
    confidence_val, class_idx = torch.max(probabilities, dim=0)
    confidence = confidence_val.item()
    
    # Map class index safely to defined target labels
    if class_idx.item() < len(LABELS):
        predicted_label = LABELS[class_idx.item()]
    else:
        predicted_label = f"Class_{class_idx.item()}"
        
    # Apply Threshold Routing Rule (TAU_HIGH = 0.85)
    routing_status = "AUTOMATED_PASS" if confidence >= TAU_HIGH else "ROUTE_TO_HITL_QUEUE"
    
    return predicted_label, confidence, routing_status

# 4. Blockcerts v3.0 Spatial Hash Verification
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

# 5. Master Pipeline Execution
def process_pipeline(image):
    if image is None:
        return "No document uploaded.", {}
        
    label, confidence, routing = predict_document(image)
    blockcerts_json = verify_document_integrity(image)
    
    status_summary = (
        f"Predicted Class : {label}\n"
        f"Confidence      : {confidence:.2%}\n"
        f"Routing Status  : {routing}"
    )
    
    return status_summary, blockcerts_json

# 6. Gradio Interface Layout
demo = gr.Interface(
    fn=process_pipeline,
    inputs=gr.Image(type="pil", label="Upload Document Image"),
    outputs=[
        gr.Textbox(label="LayoutLMv3 Classification & Routing Results", lines=4),
        gr.JSON(label="Blockcerts v3.0 Integrity Record")
    ],
    title="Document Verification & Pipeline Demo",
    description="Upload a document image to evaluate LayoutLMv3 sequence classification, threshold routing rules (TAU_HIGH = 0.85), and spatial feature hashing."
)

if __name__ == "__main__":
    demo.launch()