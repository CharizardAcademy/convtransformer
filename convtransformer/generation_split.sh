#!/bin/bash

function split_generations ()
{
  for file in `ls $1`
  do
    if [ -d $1"/"$file ]
    then
      readfile $1"/"$file
    else
      #echo $1"/"$file
      grep ^T $1"/"$file | cut -f2- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > $1"/"$file.ref
      grep ^H $1"/"$file |cut -f3- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > $1"/"$file.sys
   echo `basename $file`
   fi
  done
}

folder=`pwd`
split_generations $folder 