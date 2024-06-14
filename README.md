# Shakespeare_GPT
# Disclaimer! 
This is not original code. I followed a lecture by Andrej Karpathy who wrote the code. I followed the lecture by writing the code myself and taking notes on my work. 


# Project Summary
I trained a decoder-only model following the paper "Attention is all you need". The model is trained on the "Tiny Shakespeare" dataset (view input.txt to see dataset). The model got sensible results for my laptop, but it would have been better with a GPU. All the training code is about 200 lines of code. Architecturally speaking this code is almost identical to large GPT models like GPT3, with the biggest difference being those large models are anywhere from ten thousand to one million times larger than what I have done here.

The next step would be fine-tuning, which could look like getting a GPT in a question/answer format, getting it to perform tasks, detecting sentiment, etc. The next step could be supervised fine-tuning or something much more complex like creating a reward model to train the original model, similar to how Open AI fine-tunes their GPT.
