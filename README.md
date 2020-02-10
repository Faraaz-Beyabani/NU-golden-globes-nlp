# Golden Globes 2020 Natural Language Processing

Group Members: Faraaz Beyabani, Varun Ganglani, Raymond Liu, Brandon Lieuw

On Windows, run `win_setup.ps1` with Powershell to set up the proper virtual environment.
The script will install virtualenv, create a virtual environment, and download all of the necessary packages from the requirements.txt file.

If not on Windows, please create and activate a virtual environment (typically through virtualenv or conda), then install all necessary prerequisites like so: 

`pip install -r requirements.txt`

Make sure that all relevant and necessary JSON files are located in the same folder as gg_parser.py.

Running autograder.py will sequentially print the host(s), awards, and then the winner, presenters, and nominees for each award, followed by the best dressed and worst dressed for the whole event.

Alternatively, gg_parser.py can be run from the command line with a year argument for the same results.

After the file has been run, either in isolation or through the autograder, it will create and dump a human readable version of the results in the same folder as gg_parser.py.
This file is named 'human_readable{year}.txt'.

Repository: https://github.com/Faraaz-Beyabani/NU-golden-globes-nlp
