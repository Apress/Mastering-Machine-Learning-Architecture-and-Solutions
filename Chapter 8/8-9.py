import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# For unit testing and debugging, we import unittest and logging.
import unittest
import logging

# Set up basic logging configuration to help in debugging.
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
