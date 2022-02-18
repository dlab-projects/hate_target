import argparse
import numpy as np
import os
import pandas as pd
import pickle
import tensorflow as tf
import transformers

from hate_measure.nn.classifiers import TargetIdentityClassifier
from hate_target.utils import cv_wrapper
from hate_target import keys
from tensorflow.keras.optimizers import Adam


# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str)
parser.add_argument('--save_folder', type=str)
parser.add_argument('--save_name', type=str)
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--transformer', type=str)
parser.add_argument('--threshold', type=float, default=0.5)
parser.add_argument('--n_folds', type=int, default=5)
parser.add_argument('--val_frac', type=float, default=0.15)
parser.add_argument('--n_dense', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--max_epochs', type=int, default=20)
parser.add_argument('--dropout_rate', type=float, default=0.4)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epsilon', type=float, default=1e-8)
parser.add_argument('--early_stopping_min_delta', type=float, default=0.001)
parser.add_argument('--early_stopping_patience', type=float, default=2)
parser.add_argument('--weights', type=str, default='none')
parser.add_argument('--gpu', type=int, default=2)
args = parser.parse_args()

# Set GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
gpu = gpus[args.gpu]
tf.config.experimental.set_visible_devices(gpu, 'GPU')

# Define relevant quantities
comment_id = 'comment_id'
text_col = 'predict_text'
threshold = args.threshold

# Read in data
data = pd.read_feather(args.data_path)
comments = data[[comment_id, text_col]].drop_duplicates().sort_values(comment_id)
# Determine target identities
agreement = data[[comment_id] + keys.target_groups].groupby(comment_id).agg('mean')
is_target = (agreement >= threshold).astype('int').reset_index(level=0).merge(right=comments, how='left')
# Extract data for training models
x = is_target[text_col].values
identities = is_target[sorted(keys.target_groups)]
y = [identities[col].values.astype('int')[..., np.newaxis] for col in identities]
# Callback function
callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=args.early_stopping_min_delta,
    restore_best_weights=True,
    patience=args.early_stopping_patience)
if args.weights == 'unit':
    sample_weights = data[comment_id].value_counts().sort_index().values
elif args.weights == 'sqrt':
    sample_weights = np.sqrt(data[comment_id].value_counts().sort_index().values)
elif args.weights == 'log':
    sample_weights = 1 + np.log(data[comment_id].value_counts().sort_index().values)
else:
    sample_weights = None

transformer = args.transformer
if transformer == "distilbert-base-uncased":
    tokenizer = transformers.DistilBertTokenizer.from_pretrained(transformer)
    inputs = tokenizer(x.tolist(), return_tensors='np', padding=True)
    model_builder = TargetIdentityClassifier.build_model
    model_kwargs = {
        'transformer': transformer,
        'max_length': inputs['input_ids'].shape[1],
        'n_dense': args.n_dense,
        'dropout_rate': args.dropout_rate
    }

# Run cross-validation using Universal Sentence Encoder
(n_epochs, _, _, train_idxs, test_idxs, test_predictions, test_scores,
 model_refit, history_refit) = \
    cv_wrapper(
        x=[inputs['input_ids'], inputs['attention_mask']],
        y=y,
        model_builder=model_builder,
        model_kwargs=model_kwargs,
        compile_kwargs={'optimizer': Adam(lr=args.lr, epsilon=args.epsilon),
                        'loss': 'binary_crossentropy'},
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        n_folds=args.n_folds,
        val_frac=args.val_frac,
        refit=True,
        refit_fold=True,
        verbose=True,
        callbacks=[callback],
        cv_verbose=True,
        report_chance=False,
        unwrap_predictions=True,
        sample_weights=sample_weights)

exp_file = os.path.join(args.save_folder, args.save_name + '.pkl')
results = {
    'x': x,
    'y_true': y,
    'y_pred': test_predictions,
    'train_idxs': train_idxs,
    'test_idxs': test_idxs,
    'test_scores': test_scores,
    'n_epochs': n_epochs
}
with open(exp_file, 'wb') as results_file:
    pickle.dump(results, results_file)
model_file = os.path.join(args.save_folder, args.save_name + '_model')
model_refit.save(model_file)