# ATARI-2

Where ATARI 1 can be used as an exploratory exerise on the data, as well as traditional ML classifier techniques, ATARI 2 will aim to improve performance and use PyTorch in a systematic way, with the final goal of training the models on the DGX and getting validated surgical skill classification of over 90%. #
This is the level required before a reliable product can be made from a classificatino model. This product should be selected and toggled for different analysis types in the GUI. 


Gesture Data manipulation
- data prep - this file extracts the kinematic data and the gesture annotations, and merges them into a single file for each trial. It also creates a windowed version of the data for use in training classifiers.


Gesture classification
- Train_xgboost - this file trains an XGBoost classifier on the windowed data, and saves the model to disk. It also evaluates the model on a test set and prints the accuracy and confusion matrix.
- Train_pytorch - this file trains a PyTorch classifier on the windowed data, and saves the model to disk. It also evaluates the model on a test set and prints the accuracy and confusion matrix.
- predict_gestures - this file uses the trained model to predict gestures in new data.
- run_all - this file runs the entire pipeline, from data preparation to model training and evaluation, and saves the results to disk.

Outputs
- The outputs of the pipeline are saved to disk in the output directory specified by the user. The outputs include the trained model, the evaluation results, and the predictions on new data.





Skill classification
- S_Class_1

- ATARI 1 Classifiers
   - Old classifier models from ATARI 1

User interface
- Creat_feedback
- GUI

