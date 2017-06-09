#!/bin/bash


#First argument is mandatory username
if [ -z "$1" ]
then
       echo "Usage: $0 <username>"
       exit 1
fi


chmod -R a+r dl4cv
chmod a+x dl4cv dl4cv/classifiers
echo "Password for user $1 to upload your model files and dl4cv directory:"

rsync --delete-before -rlv -e 'ssh -x -p 58022' --exclude '*.pyc' --exclude 'output.*' models/ dl4cv $1@filecremers1.informatik.tu-muenchen.de:submit/EX3/

