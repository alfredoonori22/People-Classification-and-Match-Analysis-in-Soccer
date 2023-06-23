# People Classification and Match Analysis in Soccer
This project is developed the Computer Vision and Cognitive Systems exam of the University of Modena and Reggio Emilia.

The goal is to provide an accurate Object detector, capable of differentiate people on the field in three classes: **Player, Goalkeeper and Referee**.

And then to exploit this informations to provide some statistics starting from a football match video, like:
* the nearest player to the ball
* the distance between them
* the dominant color of his shirt
* the ball velocity

## Data used for training
* [SoccerNet](https://www.soccer-net.org/) Dataset

  SoccerNet is a large-scale dataset for soccer video understanding, it is composed of 550 complete broadcast soccer games.
After some adaptations (like removing some classes useless for our task) we used it to train the Faster-RCNN model in detecting the ball and the people on the field.
* Football People (Handcrafted)

  We extracted from the previous dataset the bounding boxes related to players, goalkeeper and referees, with their coordinates and label, and used these informations to build a balanced dataset, made of labeled images with a singol person in it.
Then this dataset was used to train the CNN in classifying a person found by the Faster-RCNN model (which prection was in this case just "Person").

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
--model: Select correct model to train (fasterrcnn or cnn), default: fastercnn

--multiclass: Select correct version of Faster-RCNN: if given, differenziate between class people: Player, Goalkeeper, Referee) else predict just Person as class

--dropout: If given add dropout layer after the two fully connected layer at the end of Fater-RCNN

--train-backbone: If given train model from scratch, without initializing backbone with default weights (trained on IMAGENET1K_V1)

# Training
--resume: If given resume from checkpoint, else start training from epoch 1

--batch-size or -b: Choose batch size, default: 4

--patience: Max number of epochs without improvements in validation score before early stopping
```

* **analysis.py options:**

    `python analysis.py`

It doesn't need training, it uses the trained model from detection task.
```bash
--deep: If given use deep model to detect players, otherwise use HogDescriptor
```

## How to get the results
* Faster-RCNN models
Test output of these models is a video, with predicted bounding boxes drawn on it frame by frame. The video ("output_detection.avi") is stored in the current directory.

* Our CNN model
Result in this case is the score (using our implementation of a "weighted" mAP) printed out in the console, and the frames with the predicted bounding boxes drawn on them are stored in a folder (which name must be "test-cnn").

* Analysis
Result is a video, with ball tracked and nearest player boxes drawn frame by frame. The distance between ball and that player, and his shirt's color, are written on the top left corner of the video. Ball velocity is also given as output in the console.
This video is stored in the current directory as "output_analysis.avi".

