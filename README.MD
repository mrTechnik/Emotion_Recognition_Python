## Usage
Project create a window application for emotion recognition via camera.
Program has two operating mods:
    1) Dynamic recognition via camera
    2) Static recognition for photo and video

This project uses:
    1) The Viola-Jones algorithm for face recognition (haarcascade_frontalface_default.xml)
    2) Conversional Neural Network (CNN) for emotion recognition (EmotionDetectionModel.h5)

```
Emotion_Recognition_Python
	├── .gitignore
	├── emotion_recognise_class.py
	├── EmotionDetectionModel.h5
	├── haarcascade_frontalface_default.xml
	├── README.MD
	└── requirements.txt
```

###Run server
The steps bellow will guide you through the project installation.  </br>
The following commands should be executed in your computer console at the project folder:

* Clone the project to your local repository workspace;

* Create a virtual environment for the imported application through the following command:
```
$ virtualenv <Virtual Environment name>
```
* Activate your virtual environment with the command:
```
$ <Virtual Environment name>\Scripts\activate
```
 * With the virtual environment properly activated, install the plugins available at the **requirements.txt** file through the command:
```
$ pip install -r requirements.txt
```
----------
Deployment
-------------
```
$ python emotion_recognise_class.py
```