# Pluvio


## 🧩 Environment Setup

**Python Version:** 3.9  
**Torch Version:** 1.10 ≤ torch < 2.3

### Install Compatible Dependencies

```bash
pip uninstall -y sentence-transformers transformers accelerate tokenizers safetensors
pip install sentence-transformers==2.2.2 \
            transformers==4.30.2 \
            accelerate==0.20.3 \
            tokenizers==0.13.3 \
            safetensors==0.3.2 \
            "torch>=1.10,<2.3"
