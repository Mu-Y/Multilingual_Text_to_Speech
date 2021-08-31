#!/bin/bash

for lang in german chinese japanese
do
    unzip -q ${lang}-single-speaker-speech-dataset.zip -d ${lang}
done
