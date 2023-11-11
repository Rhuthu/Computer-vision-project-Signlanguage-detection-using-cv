# Computer-vision-project-Signlanguage-detection-using-cv

# Features
- Real-time hand detection and tracking.
- Automatic cropping and standardization of hand sign images.
- ASL sign classification using a pre-trained deep learning model.
- Utilizes Google's Teachable Machine for model training.
- Supports custom dataset creation for further model improvement.
- Easily extensible for future enhancements and applications.

  ## Requirements
- Python 3.x
- OpenCV
- cvzone library
- Google's Teachable Machine

1. Clone the repository 
2. Install the required Python packages
 ### Image Data Collection (datacollection.py)
1. Open the terminal and navigate to the project directory.
2. Run the `datacollection.py` script to capture images of ASL signs
3. The script will open your webcam, and you can capture images by pressing the 's' key.
4. Captured images will be saved in the `Data/f` folder for creating a dataset.
   ## Dataset Creation
To train the ASL sign recognition model, you can use Google's Teachable Machine or other machine learning platforms. You will need to:
1. Create a labeled dataset using the images captured using `datacollection.py`.
2. Train the model with the dataset using your preferred machine learning platform.
3. Save the trained model to a file (e.g., `keras_model.h5`) and generate a labels file (e.g., `labels.txt`).

## Customization
You can customize the project for your specific requirements by modifying the code and configuration.
## Real-Time Sign Detection (test.py)
1. Open the terminal and navigate to the project directory.
2. Run the `test.py` script to initiate real-time ASL sign detection:
3. The application will access your webcam and display the camera feed.
4. Show ASL signs in front of your camera, and the detected sign will be displayed on the screen.

##screenshots
 Refer Project Report
#License
MIT License

Copyright (c) [2023] [RHUTHUVARNA S P]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
