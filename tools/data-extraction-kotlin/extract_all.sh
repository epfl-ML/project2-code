#!/bin/sh
echo "Extracting all files from $1 into $2"
for i in `ls $1`
do
    echo "Extracting $i"
    ./gradlew run --args="$1/$i $2/$i.csv -bin true -sleep true"
done