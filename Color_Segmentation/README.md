
# Color Segmentation
This project intend to do color segementation to detect the blue barrel from the image and obtain its segmented figures. Some of the results are shown in [Results](/Color_Segmentation/Results). <br>
![](/Color_Segmentation/Results/4_box.png)<br>
The approach to the solution is : EM Algorithm

First use Label.py to hand-labeled the barrel and label them into three labels(blue barrel, blue non-barrel and others), then using the mix_Gaussian.py to learn the model and finally show the results.
