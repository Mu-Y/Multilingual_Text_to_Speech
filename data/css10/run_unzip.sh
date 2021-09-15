#!/bin/bash

for lang in dutch
do
    unzip -q ${lang}-single-speaker-speech-dataset.zip -d ${lang}
done
