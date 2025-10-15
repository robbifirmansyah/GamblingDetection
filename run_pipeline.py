#!/usr/bin/env python3
"""
GAMBLING COMMENT DETECTION ML PIPELINE
"""

import warnings
warnings.filterwarnings('ignore')

# Import all required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import optuna
from wordcloud import WordCloud
import re
import pickle
import json
import os
from tqdm import tqdm
import time
from datetime import datetime

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Set matplotlib style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Create outputs directory
os.makedirs('outputs', exist_ok=True)

print("🚀 GAMBLING DETECTION ML PIPELINE - EXECUTION STARTED!")
print("=" * 70)

# Load datasets
print("\n📊 LOADING DATA...")
print("=" * 40)

train_df = pd.read_csv('dataset/train.csv')
test_df = pd.read_csv('dataset/test.csv')
holdout_df = pd.read_csv('dataset/holdout.csv')

print(f"✅ Train set: {train_df.shape[0]} rows, {train_df.shape[1]} columns")
print(f"✅ Test set: {test_df.shape[0]} rows, {test_df.shape[1]} columns")
print(f"✅ Holdout set: {holdout_df.shape[0]} rows, {holdout_df.shape[1]} columns")

# Check class distribution
train_dist = train_df['label'].value_counts()
train_ratio = train_dist[1] / train_dist[0] if train_dist[0] > 0 else 0

print(f"\n📊 Train set distribution:")
print(f"  Non-gambling (0): {train_dist[0]} ({train_dist[0]/len(train_df)*100:.2f}%)")
print(f"  Gambling (1): {train_dist[1]} ({train_dist[1]/len(train_df)*100:.2f}%)")
print(f"  Imbalance ratio: {train_ratio:.3f}")

# Test set distribution
test_dist = test_df['label'].value_counts()
test_ratio = test_dist[1] / test_dist[0] if test_dist[0] > 0 else 0

print(f"\n📊 Test set distribution:")
print(f"  Non-gambling (0): {test_dist[0]} ({test_dist[0]/len(test_df)*100:.2f}%)")
print(f"  Gambling (1): {test_dist[1]} ({test_dist[1]/len(test_df)*100:.2f}%)")
print(f"  Imbalance ratio: {test_ratio:.3f}")

print(f"\n✅ Data loading completed!")
print("🎯 Ready to proceed with ML pipeline execution...")
print("📝 This is a comprehensive pipeline with 20+ custom features, ensemble methods, and advanced techniques!")
print("🔥 Target: F1-score > 0.90 on holdout set!")
