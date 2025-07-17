# SageImageCaption
***

This script captions any image type using 4 models:
It uses **YOLO**, to get an object count and box the objects on the image as well as how confident it is for each object (from 0.0-100.0)
Then it uses **BLIP**, **LLava**, **Florence-2**, and **Moondream** to create 4 small captions for the image and display in the terminal.

***

## Notes:
* Initial use will take a while as you have to install all the models into your project
* After, runtime is around 40 seconds per image
* For only image captioning, you just need to run `main.py`. However, for caption evaluation using ClaudeAI, you will need to use a Claude API key which will cost credits
* Also, after testing, I found out that YOLO is not very good at detecting wildlife, but extremely accurate when detecting streets, cars, and people.

***

## Use:
- Simply run `main.py` to start the process
- After loading the models, it will ask you 3 options:
  1) selecting "1" will just generate a .json of all the generated captions for each model. This option will only take the zipfile of the images
  2) selecting "2" will evaluate existing captions only. This option will only take the .json file of the captions and your Claude API key if not preset in the code
  3) selecting "3" will do a full pipeline analysis of an image dataset. You will need to input the zipfile of images. This option will output 3 .json files: captions, evaluations, and combined report

 **Note:** For all options it may ask you to provide an output path for the .json and YOLO imageset (if you selected yes for YOLO). Simply type (anyname).json everytime it's prompted and it will auto create.

 ***
_This is ultimately for sage images that will need to process thousands of images at once, but right now that process may take an hour or so_


