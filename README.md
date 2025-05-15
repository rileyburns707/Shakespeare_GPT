# Shakespeare_GPT
# Disclaimer! 
This is not original code. I followed a lecture by Andrej Karpathy who wrote the code. I followed the lecture by writing the code myself and taking notes on my work. 


# Project Summary
This is a GPT that generates Shakespeare plays!

I trained a decoder-only model following the paper "Attention is all you need" and OpenAI's GPT-2 / GPT-3. The model is trained on the "Tiny Shakespeare" dataset (view input.txt to see dataset). The model got sensible results for my laptop, but it would have been better with a GPU. All the training code is about 200 lines of code. Architecturally speaking this code is almost identical to large GPT models like GPT3, with the biggest difference being those large models are anywhere from ten thousand to one million times larger than what I have done here.

The next step would be fine-tuning, which could look like getting a GPT in a question/answer format, getting it to perform tasks, detecting sentiment, etc. The next step could be supervised fine-tuning or something much more complex like creating a reward model to train the original model, similar to how OpenAI fine-tunes their GPTs.

# File Details
- The complete project is in the VS_Code.py file. It can be run using Visual Studio Code and will run if you have the input.txt file in the same folder. There are different hyperparameters that I wrote, I worked on a CPU so I had to lower my parameters. On a GPU the largest hyperparameters will work and the code will take roughly 15-25 minutes to completely run. The output will look exactly like the input.txt file, with the only exception being the words generated will be gibberish if run with lower hyperparameters. 

- The building_GPT.ipynb file is a Google Colab notebook that I used at the beginning of the project to understand the concepts of building a GPT. It is missing important chunks of the code such as the Multi-Head Attention module, Feed Forward module, and Block module. It was used to learn the fundamentals of how a very simple language model works. As the project got more complex Colab was too slow and clunky to use so I moved all the code to VS Code and finished it there. 

- The math_trick__for__self_attention.ipynb file is a Google Colab notebook that was a detour from the main project to fully understand self-attention. The benefits that come from making the data in lower triangular form and how linear algebra concepts apply to self-attention are discussed there.

- The input.txt file is the "Tiny Shakespeare" dataset as mentioned above. It is a simple text file of all the Shakespeare plays.

- The VS_Code_no_notes.py and the building_GPT_no_notes.py files are the exact same as the VS_Code.py and building_GPT.ipynb files with the only difference being the comments and notes. I wanted to have a place where the code was cleaned up and not as jumbled with comments everywhere, but the actual code was left untouched.

# My Thoughts on this Project
I have documented all my thoughts about what I have learned, what I struggled with, and my general ideas about the project on my LinkedIn profile here www.linkedin.com/in/riley-s-burns. I actively post and check LinkedIn so feel free to message me there about any questions you might have for me!
