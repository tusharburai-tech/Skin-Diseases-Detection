import os

bind        = f"0.0.0.0:{os.environ.get('PORT', '5000')}"
workers     = 1       # 1 worker = model loads once, saves RAM on free tier
timeout     = 300     # 5 min — enough time to download model on first cold start
preload_app = True    # load model before forking workers (avoids duplicate loads)  
