#!/bin/bash

for lang in russian
do
    unzip -q ${lang}-single-speaker-speech-dataset.zip -d ${lang}
done
