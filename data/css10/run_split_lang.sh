#!/bin/bash

for lang in german greek spanish finnish french hungarian japanese dutch russian chinese
do
    cat train.txt | grep "|${lang}|" >train_${lang}.txt
    cat val.txt | grep "|${lang}|" >val_${lang}.txt
done

