# ResiBot 2.0 – Technical Prototype
RAG + MCP Agent Architecture for 90 % communication automation at Berlin LEA.

## Quick start
```bash
git clone https://github.com/YOUR_USERNAME/resibot-2.0.git
cd resibot-2.0
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Interactive CLI
python resibot.py

# REST API
python resibot.py server   # or uvicorn resibot:app --reload
