The code is based on https://github.com/biubug6/Pytorch_Retinaface
Their code is pretty easy to use and understand. 

First download ckpt from their github, and put it in ./weights

I added a new script called inference.py, which is based on test_widerface.py and I removed 
some unnecessary components. 
I also added a script called detector.py which is based on inference.py and I wraper their code 
into a class and make this class takes in PIL.Image as input. so you can import this class from outside 
You can check use_detector.ipynb for usage

I have only tried resnet50 backbone


"dets" right above the save image in inference.py is the output information for each image 
it is a numpy with the shape of 1*15 (I only tested on image with only one face, according to 
their code, I am sure that the 1 means number of face)

For the 15, it means: box(4) + score(1) + landmarks(5*2) = 15 

Order is: 

Box (xyxy)
Score
RightEye(x,y) LeftEye(x,y) Nose(x,y) RightMouse(x,y) LeftMouse(x,y). Note that right and left means the face in the image, not from your point. 



