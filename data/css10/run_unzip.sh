#!/bin/bash

for lang in french spanish
do
    unzip -q ${lang}-single-speaker-speech-dataset.zip -d ${lang}
done
