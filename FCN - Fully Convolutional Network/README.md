# FCN - Fully Convolutional Network

In this homework, we will use all the trick at our disposal to tune a classification network on supertux. We then make this classification network fully convolutional and solve a semantic labeling task (labeling every pixel in the image).

This assignment should be solved individually. No collaboration, sharing of solutions, or exchange of models is allowed. Please, do not directly copy existing code from anywhere other than your previous solutions, or the previous master solution. We will check assignments for duplicates. See below for more details.

Tuning your classifier
In the first part of this homework, you'll tune your classifier, or the classifier of the master solution 2 to achieve as high an accuracy as possible. You may directly copy any code from your solution or the master solution to homework 2. The goal of this first part is to tune the classifier to an accuracy of . This might require several tricks:

Input normalization
Residual blocks
Dropout
Data augmentations (Both geometric and color augmentations are important. Be aggressive here. Different levels of supertux have radically different lighting.)
Weight regularization
Early stopping
...
We managed to tune our classifier to a  accuracy with a training time of 10 GPU minutes. If you manage to reach this accuracy you'll get 10 extra credits.

Test your model with

python3 -m grader homework
Relevant Operations
torchvision.transforms.Compose
torchvision.transforms.ColorJitter
torchvision.transforms.RandomHorizontalFlip
torch.nn.Dropout
and all previous
Dense prediction (semantic segmentation)
In the second part of the homework, you'll make your CNN fully convolutional. Instead of predicting a single output per image, you'll now predict an output per pixel.

Our current dataset does not support this output, we thus switch to a new dense prediction dataset. Here all images have a  resolution, and the labels are of the same size. We have 5 labels here: background, kart, track, bomb/projectile, pickup/nitro. We merged the last two classes, as they already have very few labeled instances. The class distribution for those labels is quite bad. background and track make up  of the labeled area! We will address this issue later for evaluation.

To see some examples from the dense dataset run

python3 -m homework.utils
FCN design
Design your FCN by writing the model in models.py. Make sure to use only convolutional operators, pad all of them correctly and match strided operators with up-convolutions. Use skip and residual connections.

Make sure your FCN handles an arbitrary input resolution and produces an output of the same shape as the input. Use output_padding=1 if needed. Crop the output if it is too large.

Test your model with

python3 -m grader homework
FCN Training
To train your FCN you'll need to modify your CNN training code a bit. First, you need to use the DenseSuperTuxDataset. This dataset accepts a data augmentation parameters transform. Most standard data augmentation in torchvision do not directly apply to dense labeling tasks. We thus provide you with a smaller subset of useful augmentations that properly work with a pair of image and label in dense_transforms.py.

You will need to use the same bag of tricks as for classification to make the FCN train well.

Since the training set has a large class imbalance, it is easy to cheat in a pixel-wise accuracy metric. Predicting only track and background gives a  accuracy. We additionally measure the Intersection-over-Union evaluation metric. This is a standard semantic segmentation metric that penalizes largely imbalanced predictions. This metric is harder to cheat, as it computes  \frac{\text{true positives}}{\text{true positives} + \text{false positives} + \text{false negatives}}.  You might need to change the class weights of your torch.nn.CrossEntropyLoss, although our master solution did not require this. You can compute the IoU and accuracy using the ConfusionMatrix class.

Test your model with

python3 -m grader homework
Relevant Operations
torch.optim.Adam might train faster
torch.nn.ConvTranspose2d
torch.cat for skip connections
and all previous
Grading
The test grader we provide

python3 -m grader homework -v
will run a subset of test cases we use during the actual testing. The point distributions will be the same, but we will use additional test cases. More importantly, we evaluate your model on the test set. The performance on the test grader may vary. Try not to overfit to the validation set too much.

Submission
Once you finished the assignment, create a submission bundle using

python3 bundle.py homework [YOUR UT ID]
and submit the zip file on canvas. Please note that the maximum file size our grader accepts is 20MB. Please keep your model compact. Please double-check that your zip file was properly created, by grading it again

python3 -m grader [YOUR UT ID].zip
Online grader
We will use an automated grader through canvas to grade all your submissions. There is a soft limit of 5 submisisons per assignment. Please contact the course staff before going over this limit, otherwise your submission might be counted as invalid.

The online grading system will use a slightly modified version of python and the grader:

Please do not use the exit or sys.exit command, it will likely lead to a crash in the grader
Please do not try to access, read, or write files outside the ones specified in the assignment. This again will lead to a crash. File writing is disabled.
Network access is disabled. Please do not try to communicate with the outside world.
Forking is not allowed!
print or sys.stdout.write statements from your code are ignored and not returned.
Please do not try to break or hack the grader. Doing so will have negative consequences for your standing in this class and the program.
