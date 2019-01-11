#!/bin/bash

#Create train/val/test folders
mkdir images/train
mkdir images/val
mkdir images/test

#Create train/val/test folders for each family of plane
while read family; do

    #F/A-18 family must be modified for directory name; no '/'
    if [ "$family" = "F/A-18" ]; then
        family="FA-18"
    fi

    mkdir images/train/"$family"
    mkdir images/val/"$family"
    mkdir images/test/"$family"
done < families.txt
mv families.txt images

#Move images to proper folders
folders=( train val test )
for folder in "${folders[@]}"
do
    while read record; do
        split=(${record}) #Split line on spaces
        img=${split[0]} #First element of line is the image tag
        family="${split[@]:1}" #Remaining elements are the image family

        #F/A-18 family must be modified for directory name; no '/' allowed
        if [ "$family" = "F/A-18" ]; then
            family="FA-18"
        fi

        mv images/"${img}".jpg images/${folder}/"${family}"
    done < images_family_${folder}.txt
    mv images_family_${folder}.txt images
done
