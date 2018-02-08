from django.core.management.base import BaseCommand
import websockets
import asyncio
import json

from humo.models import Molecule

import multiprocessing

# how many cores should we run this on? cpu_count will do the max
# NUM_CORES = multiprocessing.cpu_count()
NUM_CORES = 1

# fill in the path to train.csv on your computer
PATH_TO_TRAIN_FILE = "../practicals/P1-regression/train.csv"

# where to start in the dataset
START_INDEX = 0
END_INDEX = 500010

DEBUG = False

class Command(BaseCommand):

    def handle(self, *args, **options):
        """
        Watch the log files and
        """

        print ("GLOVAL VARIABLES ARE IN THE FILE", __file__)

        print ("REMEMBER TO REPLACE THE GLOBAL VARIABLES ABOVE")

        print ("THE FIRST TIME YOU RUN THIS IT SHOULD GO REALLY QUICKLY, AS YOU ARE JUST TESTING")

        # get rid of this dumb line when you've replaced the variablesk
        Molecule.load_molecules(PATH_TO_TRAIN_FILE, NUM_CORES, START_INDEX, END_INDEX, DEBUG)

        print ("MAKE DEBUG FALSE AT THE TOP RUN FOR REAL")