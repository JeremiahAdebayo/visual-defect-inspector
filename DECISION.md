# Technical decisions
This document explains the key technical decisions made during the development of Visual Defect Inspector. The goal is to make the reasoning behind each choice explicit and defensible.
## Why Patchcore
Patchcore achieves higher accuracy while cutting performance cost by a good margin 
## Why I picked resnet18 over the default wide_resnet50_2
The difference in performance between both models is negligible, just 0.01. Resnet18 offers the same performance as wide_resnet50_2 while only occupying a fraction of space. 42mb compared to wide_resnet50_2's 1.5gb. 
## Why layers 2 and 3 
Layer  2 and 3 contain feature rich maps that we're interested in. Layer 2 captures low-to-mid level features like textures, edges, and local patterns. Layer 3 captures mid-level semantic features like shapes and object parts. Together they give you both fine-grained texture information and broader structural context which is what we need to capture subtle surface defects
## Why Patchcore uses coreset subsampling and why I stuck to the default 0.1
Patchcore was trained on MVTecAD dataset extracting millions of patch embeddings. Building on all of these embeddings is not strictly necessary as we get better speed and equal performance from training on a representative subset of which 10 % is an acceptable number that doesn't trade off accuracy
## Why huggingface space over render 
Huggingface offers better space to operate than render's 500mb and is more accustomed for ML projects
## My API decision
I chose fastAPI over Flask given that Flask is synchronous, meaning it only handles one request at a time. FastAPI on the other hand is asynchronous, allowing us to free up the system in case of file processing to handle other requests. This is especially useful in our case where users need to upload image files. Additionally, fastAPI is significantly faster than Flask