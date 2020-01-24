#!/bin/bash
# WARNING NOT HEAVILY TESTED
pip install virtualenv

if [ ! -e virtual/ ]
then
    virtualenv virtual/
fi

./virtual/Scripts/activate

pip install -r ./requirements.txt