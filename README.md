# People Classification and Match Analysis in Soccer
This project is developed for the Computer Vision and Cognitive Systems exam of the University of Modena and Reggio Emilia.

The goal is to provide an accurate Object detector, capable of differentiating people on the field in three classes: **Player, Goalkeeper and Referee**.

And then to exploit these informations providing some statistics starting from a football match video, like:
* the nearest player to the ball
* the distance between them
* the dominant color of his shirt
* the ball velocity

<p align="center">
  <img width="460" height="300" src="https://github.com/chiara-cap/SoccerNet/assets/62024453/0ae859c4-fc65-47b4-8c86-f3732a89e742">
</p>

## Data used for training
* [SoccerNet](https://www.soccer-net.org/) Dataset

  SoccerNet is a large-scale dataset for soccer video understanding, it is composed of 550 complete broadcast soccer games.
After some adaptations (like removing some classes useless for our task) we used it to train the Faster-RCNN model in detecting ball and people on the field.
* Football People (Handcrafted)

  We extracted from the previous dataset the bounding boxes related to players, goalkeeper and referees, with their coordinates and label, and used these informations to build a balanced dataset, made of labeled images with just a person in it.
Then this dataset was used to train the CNN in classifying a person found by the Faster-RCNN model (which prediction was, in this case, just "Person").

## Setting up the environment
1. Create the environment
```bash
conda create -n cvcs python=3.10
```
2. Activate the environment
```bash
conda activate cvcs
```
3. Install the requirements
```bash
pip install -r requirements.txt
```

## Instructions
* **detection.py options:**
  
  `python detection.py`
```bash
--train: If you want to train the model

--test: If you want to test the model

# Model
--model: Select the model to train/test (fasterrcnn or cnn), default=fastercnn

--multiclass: Select version of Faster-RCNN. If given: differenziate between class people: Player, Goalkeeper, Referee); otherwise predict just Person as class, default=False

--dropout: If given add dropout layer after the two fully connected layer at the end of Fater-RCNN, default=False

--train-backbone: If given train model from scratch, without initializing backbone with default weights (trained on IMAGENET1K_V1), default=False

# Training
--resume: If given resume from checkpoint, otherwise start training from epoch 1, default=False

--batch-size or -b: Choose batch size, default=4

--patience: Max number of epochs without improving validation score before early stopping, default=20
```
### **Task I inference** (using our best model, with backbone's weights not initialized and dropout):

  `python detection.py --test --train-backbone --dropout`
<br><br>
* **analysis.py options:**

    `python analysis.py`

It doesn't need training, it uses the trained model from detection task.
```bash
--deep: If given use deep model to detect players, otherwise use HogDescriptor, default=False

--video-path or -v: Gives the path where is located the video used during the test, default="test.mp4" is in the current directory
```
### **Task II inference** (using our best deep model)

  `python analysis.py --deep`
<br><br>
## How to get the results
* Faster-RCNN models
  
  These models' result during test is a video, with predicted bounding boxes drawn on it frame by frame. The video ("output_detection.avi") is stored in the current directory.

* Our CNN model

  Result in this case is the score (using our implementation of a "weighted" mAP) printed out in the console, and the frames with the predicted bounding boxes drawn on them are stored in a folder (which name must be "test-cnn").

* Analysis

  Result is composed of:

  - A video, with ball tracked and nearest player boxes drawn frame by frame. The distance between ball and that player, and his shirt's color, are written on the top left corner of the video. Ball velocity is also given as output in the console.
This video is stored in the current directory as "output_analysis.avi".
  - An image, which represents the ball possession percentage of the two teams, in a pie chart. This image is stored in the current directory as "possession.jpeg".

