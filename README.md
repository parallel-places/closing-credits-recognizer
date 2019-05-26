# Closing Credits Recognizer

A tool using deep learning for detecting when the credits would start rolling at the end of movies or series. Being able to automatically tell when the closing credits would start at the end has different use cases. For example, in streaming services you can give the user the possibility to move to the next episode or show them recommendations. A case you might have encountered is in Netflix, where when the closing credits start netflix is going to start the next episode automatically unless you select not to.

## Project structure and information

Currently repository has two main parts. First, the notebooks directory where the feasibility of the idea by exploring whether it is possible to tell the closing credits screenshots apart from normal movie screenshots (approaching the problem as a binary classification) and processing video files. Second, script directory has a python script to detect the closing credits' starting point locally. In addition, the data used to train models is provided in the data directory. It is planned to deploy the model as an API as the next step.

### Dependencies

Project uses Python 3.6 and Pipenv to manage the dependencies. If you're going to install the dependencies using Pipenv and run the notebooks or the script on a computer without GPU, I suggest to delete the ```Pipfile.lock``` and change the line ```tensorflow-gpu = "*"``` in the Pipfile to ```tensorflow = "*"```.

### Pretrained model

In case you don't have access to a GPU or can't train the model yourself, the trained Keras Resnet50 model for this project is available for download from [here (Google Cloud Storage)](https://storage.googleapis.com/parallel_places/closing_credits_Resnet50.h5). By downloading the trained model you can easily run the script and as output get the frame ID and the time when the credits start rolling.

### How to run

After downloading the pretrained model and installing the dependencies you can easily test the pipeline by running the script ```closing_credit_recognizer.py``` and give it the following command line arguments:

```
python closing_credit_recognizer.py path/to/video.mp4 path/to/model/closing_credits_Resnet50.h5
```

For example, I have downloaded "Sintel" movie (an open source movie) from [here](https://durian.blender.org/download/) and ran the script:

```
python closing_credit_recognizer.py sintel-2048-surround.mp4 closing_credits_Resnet50.h5
```

which would output the following, printing the frame number where credits started and its exact time:

```
Credits started rolling at 00:12:24.0, at 17856.0 frame.
```