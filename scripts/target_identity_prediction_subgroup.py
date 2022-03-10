import argparse
import numpy as np
import os
import pandas as pd
import pickle
import tensorflow as tf
import transformers

from hate_measure.nn import classifiers
from hate_target.datasets import load_subgroup
from hate_target.utils import cv_wrapper
from hate_target import keys
from tensorflow.keras.optimizers import Adam


# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str)
parser.add_argument('--save_folder', type=str)
parser.add_argument('--save_name', type=str)
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--subgroup', type=str, default='race')
parser.add_argument('--threshold', type=float, default=0.5)
parser.add_argument('--n_folds', type=int, default=5)
parser.add_argument('--val_frac', type=float, default=0.15)
parser.add_argument('--model', type=str, default='use_v5')
parser.add_argument('--n_dense', type=int, default=128)
parser.add_argument('--pooling', type=str, default='max')
parser.add_argument('--mask_pool', action='store_true')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--max_epochs', type=int, default=20)
parser.add_argument('--dropout_rate', type=float, default=0.4)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epsilon', type=float, default=1e-8)
parser.add_argument('--early_stopping_min_delta', type=float, default=0.001)
parser.add_argument('--early_stopping_patience', type=float, default=2)
parser.add_argument('--weights', type=str, default='none')
parser.add_argument('--soft', action='store_true')
parser.add_argument('--gpu', type=int, default=-1)
args = parser.parse_args()

# Deal with environment settings
os.environ["TFHUB_CACHE_DIR"] = os.path.join(os.environ['HOME'], '.cache/tfhub_modules')
# Limit GPU memory usage
gpus = tf.config.experimental.list_physical_devices('GPU')
[tf.config.experimental.set_memory_growth(gpu, True) for gpu in gpus]
# Request specific GPU
if args.gpu != -1:
    gpu = gpus[args.gpu]
    tf.config.experimental.set_visible_devices(gpu, 'GPU')

# Define relevant quantities
comment_id = 'comment_id'
text_col = 'predict_text'
threshold = args.threshold

# Read in data
data, x, y_hard, y_soft = load_subgroup(args.data_path, group=args.subgroup)

if args.soft:
    y_true = y_soft
else:
    y_true = y_hard
# Assign weights to samples
if args.weights == 'unit':
    sample_weights = data[comment_id].value_counts().sort_index().values
elif args.weights == 'sqrt':
    sample_weights = np.sqrt(data[comment_id].value_counts().sort_index().values)
elif args.weights == 'log':
    sample_weights = 1 + np.log(data[comment_id].value_counts().sort_index().values)
else:
    sample_weights = None
# Create callback function
callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=args.early_stopping_min_delta,
    restore_best_weights=True,
    patience=args.early_stopping_patience)
# Create model parameters
model = args.model
if model == "distilbert-base-uncased":
    tokenizer = transformers.DistilBertTokenizer.from_pretrained(model)
    inputs = tokenizer(x.tolist(), return_tensors='np', padding=True)
    model_builder = classifiers.MultiBinaryClassifier.build_model
    model_kwargs = {
        'outputs': sorted(keys.target_race_cols),
        'transformer': model,
        'max_length': inputs['input_ids'].shape[1],
        'n_dense': args.n_dense,
        'dropout_rate': args.dropout_rate
    }
elif model == "bert-base-uncased":
    tokenizer = transformers.BertTokenizer.from_pretrained(model)
    tokens = tokenizer(x.tolist(), return_tensors='np', padding=True)
    inputs = [tokens['input_ids'], tokens['attention_mask']]
    model_builder = classifiers.MultiBinaryClassifier.build_model
    model_kwargs = {
        'outputs': sorted(keys.target_race_cols),
        'transformer': model,
        'max_length': tokens['input_ids'].shape[1],
        'n_dense': args.n_dense,
        'dropout_rate': args.dropout_rate,
        'pooling': args.pooling,
        'mask_pool': args.mask_pool
    }
# Create compile arguments
compile_kwargs = {
    'optimizer': Adam(lr=args.lr, epsilon=args.epsilon),
    'loss': 'binary_crossentropy'
}

# Run cross-validation using Universal Sentence Encoder
cv_results = cv_wrapper(
        x=inputs,
        y=y_true,
        model_builder=model_builder,
        model_kwargs=model_kwargs,
        compile_kwargs=compile_kwargs,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        n_folds=args.n_folds,
        val_frac=args.val_frac,
        refit=False,
        refit_fold=True,
        verbose=True,
        callbacks=[callback],
        cv_verbose=True,
        unwrap_predictions=True,
        store_models=False,
        sample_weights=sample_weights)

exp_file = os.path.join(args.save_folder, args.save_name + '.pkl')
results = {
    'x': x,
    'y_true': y_true,
    'y_soft': y_soft,
    'y_hard': y_hard,
    'y_pred': cv_results['test_predictions'],
    'train_idxs': cv_results['train_idxs'],
    'test_idxs': cv_results['test_idxs'],
    'test_scores': cv_results['test_scores'],
    'n_epochs': cv_results['n_epochs']
}
with open(exp_file, 'wb') as results_file:
    pickle.dump(results, results_file)
if 'model_refit' in results:
    model_file = os.path.join(args.save_folder, args.save_name + '_model.h5')
    cv_results['model_refit'].save_weights(model_file)
