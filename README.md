#Brawl Stars AI

This repo implements a gym-like environment for Brawl Stars application.
The implementation key-words are:
- BlueStacks
- screen capturing
- mouse and keyboard control
- real-time static ocr

Refer to https://github.com/MikeTkachuk/Brawl_iris for modelling implementation.


####Ideas:
**Speed-up:**
- try pretraining tokenizer and world model
- fp16 precision
- smaller batch size
- use an ~2 times cheaper g4dn.4xlarge (0.53 vs 0.30 per GPU)

**Results**
- try working with optical flow (just a crazy idea)