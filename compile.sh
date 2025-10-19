#!/bin/bash

# Базовый скрипт компиляции
SOURCE_FILE="main.cpp"
OUTPUT_FILE="program"

g++ -o $OUTPUT_FILE $SOURCE_FILE -O2 -fopenmp

echo "имя исполняемго файла - program"
echo "порядок аргументов - M N K P"