#!/bin/bash
chmod -R a+r dl4cv
chmod a+x dl4cv dl4cv/classifiers
echo "Password for user $1 to upload your model files and dl4cv directory:"
echo "put models/* 
put -rp dl4cv" | sftp -P 58022 $1@filecremers1.informatik.tu-muenchen.de:/$1/submit/EX2/

