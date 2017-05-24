#!/bin/bash
echo "Password for user $1 to upload your model files and dl4cv directory:"
echo "put models/* 
put -r dl4cv" | sftp -P 58022 $1@filecremers1.informatik.tu-muenchen.de:/$1/submit/EX1/

