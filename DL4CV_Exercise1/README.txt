Deep Learning for Computer Vision - Technical University Munich - Summer 2017

In this file we will describe:
1) Our Python setup
2) Getting the data
2) Exercise submission

We want to thank the Stanford Vision Lab for allowing us to build these
exercises on material they had previously developed. 

############################
Python setup:

For this exercise we use Python 2.7. For the following description, we assume that you are using Linux or
MacOS and that you are familar with working from a terminal. 
If you are using Windows, the procedure might slightly vary and you will have to google for the details.

We highly recommend setting up a Python virtual environment for the programming exercises of this lecture. 
To this end, install or upgrade virtualenv. There are several ways depending
on your OS. At the end of the day, we want 

which virtualenv

to point to the installed location.

On Ubuntu, you can use: apt-get install python-virtualenv

Also, installing with pip should work, if you subsequently put the executable
in your search path:
pip install virtualenv

Once virtualenv is successfully installed, create a directory, say
dl4cv_exercises. We can use the same virtual environment for all exercises, so
make dl4cv_exercises a parent directory of this folder. Then execute:
cd dl4cv_exercises 
virtualenv --no-site-packages .env

Basically, this installs a sandboxed Python in the directory .env. The
additional argument ensures that sandboxed packages are used even if they had
already been installed globally before.

Whenever you want to use this virtualenv in a shell you have to first
activate it via calling (from dl4cv_exercises). Do that now:
source .env/bin/activate

To test whether your virtualenv activation has worked, call:
which python

This should now point to .env/bin/python.

From now on we assume that that you have activated your virtualenv.


Installing required packages:
We have made it easy for you to get started, just call from within the
exercise folder:
pip install -r requirements.txt


The exercises are guided via jupyter notebooks. Open such a notebook in your
browser (we have tested on Chrome) from the exercise root folder by executing:
jupyter notebook

Now you can get starting by selecting the respective exercise!

If you are looking for a good Python development environment, we recommend
PyCharm Community Edition, which you can use free of charge.


############################
Getting the data:
To download the data, execute the script get_datasets.sh in datasets/.
You will need ~400MB of disk space.


############################
Exercise submission:
After completing the exercises you will be submitting trained models to be
automatically evaluated on a test set on our server.

To this end, login or register for an account at:
https://vision.in.tum.de/teaching/ss2017/dl4cv/submit

Note that only students, who have registered for this class in TUM Online can
register for an account.
This account provides you with temporary credentials to login onto the
machines at our chair. You may use the computers in room 02.05.14, but you can
also work from your own computer.

After you have worked through the exercises, your saved models will be in the
models/ subfolder of your local working directory for this excercise. 
In order to submit the models, you have to (directly, not
in in a subfolder) put them in the folder ~/submit/{EX1, EX2, EX3} on your
account, which you have created. Don't change their filenames!
Additionally, you have to put your dl4cv directory in ~/submit/{EX1, EX2, EX3}
as well.
You can do everything automatically by using the submit script in the exercise root directory. 
Call it like:
./submit_exerciseX.sh s999

where X={1,2,3} for the respective exercise and s999 has to be substituted by your
username in our system.
This script uses sftp to transfer the models onto our lab's machine. Make sure
sftp is installed, i.e. calling sftp in a terminal should give some output.

Once the models (python pickle files) are in ~/submit/{EX1, EX2, EX3}, you can
login to the above website, where they can be selected for submission.
Note that you have to explicitly submit the files through the webinterface,
just uploading them to the respective directory is not enough. 

You will receive an email notification with the results upon completion of the
evaluation.

To make it more fun, you will be able to see a leaderboard of everyone's
(anonymous) scores on the login part of the submission website.

Note that you can re-evaluate your models until the deadline. Whereas the
email contains the result of the current evaluation, the entry in the
leaderboard always represents the best score for the respective exercise.
