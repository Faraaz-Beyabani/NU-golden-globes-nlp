# Golden Globes 2020 Natural Language Processing

Group Members: Faraaz Beyabani, Varun Ganglani, Raymond Liu, Brandon Lieuw

On Windows, run `win_setup.ps1` with Powershell to set up the proper virtual environment.
The script will install virtualenv, create a virtual environment, and download all of the necessary packages from the requirements.txt file.

If not on Windows, please activate the `virtual` virtual environment by running ./virtual/Scripts/activate.bat, then installing all necessary prerequisites like so: 

`pip install -r requirements.txt`

Make sure that all relevant and necessary JSON files are located in the ./data/ folder.

Running autograder.py will sequentially print the host(s), awards, and then the winner, presenters, and nominees for each award, followed by the best dressed and worst dressed for the whole event.

Alternatively, gg_parser.py can be run from the command line with a year argument for the same results.

Repository: https://github.com/Faraaz-Beyabani/NU-golden-globes-nlp