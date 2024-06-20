<h1>Colour Recognition using Python</h1>

<h3>Area / Domain of Project:</h3>
Artificial Intelligence and Machine Learning

<h3>Abstract:</h3>
In response to the growing importance of color recognition in various applications,  
this study delves into the realm of machine learning to address the challenges
associated with accurate and efficient color recognition. The existing methods for 
color recognition often face limitations in terms of adaptability to diverse 
environments and robustness against variations in lighting conditions. The aim is 
to enhance the precision and versatility of color recognition systems, filling the gap 
in current methodologies. By leveraging machine learning algorithms, particularly 
deep learning models, this research intends to develop a more sophisticated and 
adaptive color recognition system capable of handling real-world scenarios. The 
significance lies in the broader applications, ranging from computer vision and 
image processing to assistive technologies and product quality control. Addressing 
these challenges not only contributes to the advancement of color recognition 
technology but also facilitates progress in various fields where accurate color 
identification is pivotal. 
<hr>
The project on color recognition using OpenCV involves developing a system that can identify and distinguish different colors in images or video streams. OpenCV (Open Source Computer Vision Library) provides a comprehensive set of tools and functions for image processing and computer vision tasks.


<h3>Key Components:</h3>

Image Acquisition: 
Capture video frames using a camera or load them from files.

Color Space Conversion:
Convert images from the default BGR (Blue, Green, Red) color space to other color spaces like HSV (Hue, Saturation, Value) which is more suitable for color segmentation.

Color Detection: 
Define the color ranges for the colors to be recognized. This is typically done by setting the lower and upper bounds for the HSV values of each color.

Mask Creation: 
Create masks for each color range, isolating the pixels that fall within the specified HSV ranges.

Contour Detection:
Identify and extract the contours of the colored regions from the masks. Contours are curves joining all the continuous points along the boundary of a color.

Drawing Boundaries and Labels:
Draw contours and bounding boxes around the detected colored regions and label them accordingly.

<h3>Applications:</h3>

Object tracking

Traffic light detection

Color-based sorting systems

Interactive art installations

This project demonstrates the practical use of computer vision techniques in real-time applications, making it a fundamental exercise in the field of image processing and pattern recognition.
Display Results:
Show the original video feed with the detected colors highlighted and labeled in real-time.
