#Brawl Stars AI

This repo implements a gym-like environment for Brawl Stars application.
The implementation key-words are:
- BlueStacks
- screen capturing
- mouse and keyboard control
- real-time static ocr

Refer to https://github.com/MikeTkachuk/Brawl_iris for modelling implementation.


TODO: 
- ~~reach 2 it per second for agent~~
- ~~efficiently upload artifacts to aws~~
- component cloud training + loop checkpoint funneling (upload, train, download, collect)
- configure aws ec2 and batch (quota, test autoscale, benchmark scale up/down and pricing, ~~configure IAM for jobs~~)
- try pretraining tokenizer
- try working with optical flow (just a crazy idea)