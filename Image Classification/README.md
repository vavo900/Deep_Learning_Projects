# Image Classification

In this homework, we will train a simple deep network to classify images from SuperTuxKart.

This assignment should be solved individually. No collaboration, sharing of solutions, or exchange of models is allowed. Please, do not directly copy existing code from anywhere other than your previous solutions, or the previous master solution. We will check assignments for duplicates. See below for more details.

Running your assignment on google colab
You might need a GPU to train your models. You can get a free one on google colab. We provide you with a ipython notebook that can get you started on colab for each homework.

If you've never used colab before, go through colab notebook (tutorial)
When you're comfortable with the workflow, feel free to use colab notebook (shortened)

Follow the instructions below to use it.

Go to http://colab.research.google.com/.
Sign in to your Google account.
Select the upload tab then select the .ipynb file.
Follow the instructions on the homework notebook to upload code and data.
Starter code and dataset
The starter code for this assignment can be found here. The starter code contains several useful scripts:

bundle.py will zip up your homework, ready for submission. Only submit zip files created by this bundling script
grader locally grades your submission, works with both zip files and your homework directory.
The starter code also contains a data directory where you'll copy (or symlink) the SuperTuxKart classification dataset. Unzip the data directly into the homework folder, replacing the existing data directory completely. Make sure you see the following directories and files inside your main directory

homework
grader
bundle.py
data
data/train
data/valid
You will run all scripts from inside this main directory.

Data Loader (15 pts)
As a first step, we will need to implement a data loader for the SuperTuxKart dataset. Complete the __init__, __len__, and the __getitem__ of the SuperTuxDataset class in the utils.py.

The __len__ function should return the size of the dataset.

The __getitem__ function should return a tuple of image, label. The image should be a torch.Tensor of size (3,64,64) with range [0,1], and the label should be int.

Labels and the corresponding image paths are saved in labels.csv, their headers are file and label. There are 6 classes of objects. Make sure label background corresponds to 0, kart is 1, pickup is 2, nitro is 3, bomb is 4 and projectile 5.

Once you finish, you can visualize some of the images by their classes using:

python3 -m homework.visualize_data data/valid
viz

Hint: We recommend using the csv package to read csv files and the PIL library (Pillow fork) to read images in Python.

Hint: Use torchvision.transforms.ToTensor() to convert the PIL image to a pytorch tensor.

Hint: You have (at least) two options on how to load the dataset. You can load all images in the __init__ function, or you can lazily load them in __getitem__. If you load all images in __init__, make sure you convert the image to a tensor in the constructor, otherwise, you might get an OSError: [Errno 24] Too many open files.

Congratulations, you finished the first part. Time to grade your data loader.

python3 -m grader homework -v
Relevant Operations
torchvision.transforms.ToTensor
torch.utils.data.Dataset
csv.reader
PIL.Image.open
Linear Model (15 pts)
Implement the LinearClassifier class in models.py. Define the linear model and all layers in the __init__ function, then implement forward. Your forward function receives a (B,3,64,64) tensor as an input and should return a (B,6) torch.Tensor (one value per class). You can earn these full credits without training the model, just from the correct model definition.

You can grade your linear model using

python3 -m grader homework -v
Relevant Operations
torch.nn.Linear
torch.tensor.View
and all previous
Classification Loss (10 pts)
Next, we'll implement the ClassificationLoss in models.py. We will later use this loss to train our classifiers. You should implement the log-likelihood of a softmax classifier.

where  are the logits and  is the label. You may use existing pytorch functions to implement this.

You can grade your loss using

python3 -m grader homework -v
Relevant Operations
torch.nn.functional
and all previous
Training the linear model (30 pts)
Train your linear model in train.py. You should implement the full training procedure

create a model, loss, optimizer
load the data: train and valid
Run SGD for several epochs
Save your final model, using save_model
You can train your network using

python3 -m homework.train -m linear
Hint: You might find it useful to store optimization parameters in the ArgumentParser, and quickly try a few from the command-line.

Hint: You might find it useful to allow training of an existing model to continue. Use the torch.load function for that. See load_model.

Hint: Try to write your training code model agnostic. We will swap out the model below.

We will use this checkpoint to grade your linear model's performance. You can grade your trained model using

python3 -m grader homework -v
Relevant Operations
torch.optim.Optimizer
torch.optim.SGD
torch.Tensor.backward
and all previous
MLP Model (30 pts)
Implement the MLPClassifier class in models.py. The inputs and outputs to the multi-layer perceptron are the same as the linear classifier. However, now you're learning a non-linear function.

You can train your network using

python3 -m homework.train -m mlp
Hint: This part might require some tuning of your training code. Try to move most modifications to command-line arguments in ArgumentParser

Hint: Use ReLU layers as non-linearities.

Hint: Two layers are sufficient.

Hint: Keep the first layer small to save parameters.

You can test your trained model using

python3 -m grader homework -v
Relevant Operations
torch.nn.ReLU
torch.nn.Sequential
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