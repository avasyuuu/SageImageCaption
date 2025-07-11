# SageImageCaption

This script captions any image type using 4 models:
It uses YOLO, to get an object count and box the objects on the image as well as how confident it is for each object (from 0.0-100.0)
Then it uses BLIP, LLava, Florence-2, and Paligemma to create 4 small captions for the image and display in the terminal.

Notes:
Initial use will take a while as you have to install all the models into your project
After, runtime is around 1 min 30 seconds
Every model works great except for Paligemma where you have to do some extra things:
Paligemma is not complete public access and you ahve to get an access token from huggingface. To  do this you ahve to create a huggface accounts and run
"huggingface-cli login"
It will then prompt you insert an access token which you will create in huggingface.

For use, simplly put the image path under "if main == "__main__":

you can also do "batch" images with multiple urls under that.
