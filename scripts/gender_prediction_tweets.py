import h5py
import numpy as np
import tensorflow as tf

from hate_measure.nn.classifiers import MultiBinaryClassifier

gpus = tf.config.experimental.list_physical_devices('GPU')
gpu = gpus[0]
tf.config.experimental.set_visible_devices(gpu, 'GPU')
tf.config.experimental.set_memory_growth(gpu, True)

token_path = "/home/psachdeva/data/twitter_tokenized.h5"
weights_path = '/home/psachdeva/experiments/target_gender/gender_exp01e_model.h5'
save_path = "/home/psachdeva/data/gender_twitter_predictions.h5"
batch_size = 8

print('Reading in tokens...')
with h5py.File(token_path, 'r') as file:
    input_ids = file['input_ids'][:]
    attention_mask = file['attention_mask'][:]

target_cols = [
    'target_gender_men',
    'target_gender_non_binary',
    'target_gender_transgender',
    'target_gender_women']
model_kwargs = {
    'outputs': target_cols,
    'transformer': 'bert-base-uncased',
    'max_length': 247,
    'n_dense': 64,
    'dropout_rate': 0.1,
    'pooling': 'mean',
    'mask_pool': False
}
print('Creating model...')
model_builder = MultiBinaryClassifier.build_model
model = model_builder(**model_kwargs)
model.load_weights(weights_path)

print('Running predictions...')
y_pred = model.predict(
    [input_ids, attention_mask],
    batch_size=batch_size,
    verbose=1)

print('Saving predictions...')
with h5py.File(save_path, 'w') as file:
    file['y_pred'] = np.squeeze(np.array(y_pred)).T
