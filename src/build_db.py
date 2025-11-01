import os, json
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CocoCaptions
from torchvision import transforms
import chromadb
import numpy as np
import open_clip
# from transformers import OFATokenizer, OFAModel

# 3️⃣ Paths
COCO_IMG_DIR = "./coco/train2017"
COCO_ANN_FILE = "./coco/annotations/captions_train2017.json"
assert os.path.exists(COCO_IMG_DIR)
assert os.path.exists(COCO_ANN_FILE)

# 4️⃣ Persistent Chroma setup
chroma_dir = "./chroma_db"
client = chromadb.PersistentClient(path=chroma_dir)
# Create or get collection
if "coco_clip_embeddings" in [c.name for c in client.list_collections()]:
    collection = client.get_collection("coco_clip_embeddings")
else:
    collection = client.create_collection("coco_clip_embeddings")

# 5️⃣ Load CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
clip_model = clip_model.to(device)
clip_model.eval()

# 6️⃣ Load COCO annotations
with open(COCO_ANN_FILE, "r") as f:
    coco_data = json.load(f)

annotations = coco_data["annotations"]
img_id_to_path = {int(fn.split('.')[0]): os.path.join(COCO_IMG_DIR, fn)
                  for fn in os.listdir(COCO_IMG_DIR) if fn.endswith(".jpg")}

# 7️⃣ Encode images & store embeddings + captions
batch_embeds = []
batch_docs = []
batch_ids = []

print("🔹 Encoding images and indexing in Chroma...")
for ann in tqdm(annotations):  # demo limit
    img_id = ann["image_id"]
    ann_id = ann["id"]
    caption = ann["caption"]
    img_path = img_id_to_path.get(img_id)
    if not os.path.exists(img_path):
        continue

    # Encode image
    image = clip_preprocess(Image.open(img_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        img_emb = clip_model.encode_image(image)
        img_emb /= img_emb.norm(dim=-1, keepdim=True)
        emb = img_emb.cpu().numpy()[0]

    batch_embeds.append(emb)
    batch_docs.append(caption)
    batch_ids.append(f"{img_id}_{ann_id}")  # unique ID

    if len(batch_embeds) == 256:
        collection.add(
            embeddings=np.array(batch_embeds).tolist(),
            documents=batch_docs,
            ids=batch_ids
        )
        batch_embeds, batch_docs, batch_ids = [], [], []

# Add remaining
if batch_embeds:
    collection.add(
        embeddings=np.array(batch_embeds).tolist(),
        documents=batch_docs,
        ids=batch_ids
    )


print("✅ Chroma database saved.")