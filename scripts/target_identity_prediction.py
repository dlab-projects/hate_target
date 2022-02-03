import argparse
import numpy as np
import os
import pandas as pd
import pickle
import tensorflow as tf

from hate_measure.nn.classifiers import TargetIdentityClassifierUSE
from hate_measure.utils import cv_wrapper
from hate_target import keys
from tensorflow.keras.optimizers import Adam


# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str)
parser.add_argument('--save_folder', type=str)
parser.add_argument('--save_name', type=str)
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--agreement', type=float, default=0.5)
parser.add_argument('--n_folds', type=int, default=5)
parser.add_argument('--val_frac', type=float, default=0.15)
parser.add_argument('--use_version', type=str, default='v5')
parser.add_argument('--n_dense', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--max_epochs', type=int, default=20)
parser.add_argument('--dropout_rate', type=float, default=0.4)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epsilon', type=float, default=1e-8)
parser.add_argument('--early_stopping_min_delta', type=float, default=0.001)
parser.add_argument('--early_stopping_patience', type=float, default=2)
parser.add_argument('--gpu', type=int, default=2)
args = parser.parse_args()

# Deal with environment settings
os.environ["TFHUB_CACHE_DIR"] = os.path.join(os.environ['HOME'],
                                             '.cache/tfhub_modules')
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
identities = agreement[sorted(keys.target_groups)]
y = [identities[col].values.astype('int')[..., np.newaxis] for col in identities]
# Callback function
callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=args.early_stopping_min_delta,
    restore_best_weights=True,
    patience=args.early_stopping_patience)

# Run cross-validation using Universal Sentence Encoder
(n_epochs, _, _, train_idxs, test_idxs, test_predictions, test_scores,
 model_refit, history_refit, chance) = \
    cv_wrapper(
        x=[x],
        y=y,
        model_builder=TargetIdentityClassifierUSE.build_model,
        model_kwargs={'n_dense': args.n_dense,
                      'version': args.use_version,
                      'dropout_rate': args.dropout_rate},
        compile_kwargs={'optimizer': Adam(lr=args.lr, epsilon=args.epsilon),
                        'loss': 'categorical_crossentropy',
                        'metrics': ['acc']},
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        n_folds=args.n_folds,
        val_frac=args.val_frac,
        refit=True,
        refit_fold=True,
        verbose=True,
        callbacks=[callback],
        cv_verbose=True,
        report_chance=True,
        unwrap_predictions=True)

exp_file = os.path.join(args.save_folder, args.save_name + '.pkl')
with open(exp_file, 'wb') as results:
    pickle.dump([n_epochs, train_idxs, test_idxs, test_predictions, test_scores,
                 chance], results)
model_file = os.path.join(args.save_folder, args.save_name + '_model.h5')
model_refit.save_weights(model_file)
