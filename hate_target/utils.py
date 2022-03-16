import ftfy
import gc
import multiprocessing
import numpy as np
import pickle
import tensorflow as tf

from arrayqueues.shared_arrays import ArrayQueue
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score
from tensorflow.keras.optimizers import Adam
from textacy import preprocessing as pp


def analyze_experiment(path, soft=False, verbose=True):
    """Analyze target identity experiment results.

    Parameters
    ----------
    path : string
        The path to the results file.
    n_groups : int
        The number of groups in the task.

    Returns
    -------
    results : dict
        The results dictionary.
    """
    with open(path, 'rb') as file:
        results = pickle.load(file)

    test_scores = results['test_scores']
    if soft:
        y_true = results['y_hard']
    else:
        y_true = results['y_true']
    y_pred = results['y_pred']
    train_idxs = results['train_idxs']
    test_idxs = results['test_idxs']

    overall_loss = test_scores[:, 0]
    label_loss = test_scores[:, 1:]
    accuracy_by_chance = cv_multilabel_accuracy_by_chance(
        y_true, y_pred, train_idxs, test_idxs)
    log_odds_difference = cv_multilabel_log_odds_difference(
        y_true, y_pred, train_idxs, test_idxs)
    roc_aucs = cv_multilabel_metric(
        y_true, y_pred, test_idxs, 'roc_auc')
    pr_aucs = cv_multilabel_metric(
        y_true, y_pred, test_idxs, 'pr_auc')

    if verbose:
        print(f"Overall loss: {overall_loss.mean():0.4f}")
        print(f"Label loss: {label_loss.mean():0.4f}")
        print(f"Accuracy by chance: {accuracy_by_chance.mean():0.4f}")
        print(f"Log-odds difference: {log_odds_difference.mean():0.4}")
        print(f"ROC AUC: {roc_aucs.mean():0.4f}")
        print(f"PR AUC: {pr_aucs.mean():0.4f}")

    analysis = {
        'overall_loss': overall_loss,
        'label_loss': label_loss,
        'accuracy_by_chance': accuracy_by_chance,
        'log_odds_difference': log_odds_difference.T,
        'roc_aucs': roc_aucs.T,
        'pr_aucs': pr_aucs.T
    }
    return analysis


def accuracy_by_chance(p_train, p_test):
    """Calculates the accuracy by randomly assigning labels on a test set,
    using statistics from the training set.

    This function assumes binary classification.

    Parameters
    ----------
    p_train, p_test : float
        The fraction of samples in the positive class on the training set and
        test set, respectively.
    """
    return 2 * p_train * p_test - p_train - p_test + 1


def log_odds_difference(accuracy, chance):
    """Calculates the log-odds difference between an accuracy and chance.

    Parameters
    ----------
    accuracy : float
        The accuracy on the test set.
    chance : float
        The chance on the test set.
    """
    return np.log(accuracy / (1 - accuracy)) - np.log(chance / (1 - chance))


def cv_accuracy_by_chance(y_true, y_pred, train_idxs, test_idxs, threshold=0.5):
    """Calculates the accuracy by chance over CV folds.

    Parameters
    ----------
    y_true : array-like
        The true labels, with first dimension denoting CV fold and second
        dimension denoting samples.
    y_pred : array-like
        The predicted probabilities of each label, with same dimensions as
        y_true.
    train_idxs : list of arrays
        The training indices for each fold.
    test_idxs : list of arrays
        The test indices for each fold.
    threshold : float
        The threshold by which to convert predicted probabilities to labels.

    Returns
    -------
    accuracy_by_chances : np.ndarray
        An array with the accuracy by chance on each fold.
    """
    n_folds = len(train_idxs)
    accuracy_by_chances = np.zeros(n_folds)

    for idx, (train_idx, test_idx) in enumerate(zip(train_idxs, test_idxs)):
        # Obtain train and test samples
        y_true_train = y_true[train_idx].ravel()
        y_true_test = y_true[test_idx].ravel()
        y_pred_test = (y_pred[test_idx] >= threshold).astype('int').ravel()
        # Calculate chance
        p_train = y_true_train.mean()
        p_test = y_true_test.mean()
        chance = accuracy_by_chance(p_train, p_test)
        # Calculate accuracy
        accuracy = np.mean(y_pred_test == y_true_test)
        # Accuracy over chance
        accuracy_by_chances[idx] = accuracy / chance

    return accuracy_by_chances

def cv_log_odds_difference(y_true, y_pred, train_idxs, test_idxs, threshold=0.5):
    """Calculates the log-odds difference over CV folds.

    Parameters
    ----------
    y_true : array-like
        The true labels across all samples.
    y_pred : array-like
        The predicted probabilities of each label, with same dimensions as
        y_true.
    train_idxs : list of arrays
        The training indices for each fold.
    test_idxs : list of arrays
        The test indices for each fold.
    threshold : float
        The threshold by which to convert predicted probabilities to labels.

    Returns
    -------
    log_odds_differences : np.ndarray
        An array with the log-odds difference on each fold.
    """
    n_folds = len(train_idxs)
    log_odds_differences = np.zeros(n_folds)

    for idx, (train_idx, test_idx) in enumerate(zip(train_idxs, test_idxs)):
        # Obtain train and test samples
        y_true_train = y_true[train_idx].ravel()
        y_true_test = y_true[test_idx].ravel()
        y_pred_test = (y_pred[test_idx] >= threshold).astype('int').ravel()
        # Calculate chance
        p_train = y_true_train.mean()
        p_test = y_true_test.mean()
        chance = accuracy_by_chance(p_train, p_test)
        # Calculate accuracy
        accuracy = np.mean(y_pred_test == y_true_test)
        # Accuracy over chance
        log_odds_differences[idx] = log_odds_difference(accuracy, chance)

    return log_odds_differences

def cv_multilabel_accuracy_by_chance(y_true, y_pred, train_idxs, test_idxs, threshold=0.5):
    """Calculates the accuracy by chance over CV folds and labels.

    Parameters
    ----------
    y_true : list of array-like
        The true labels, as a list of arrays. The list is over labels, and the
        array is over all samples.
    y_pred : array-like
        The predicted probabilities of each label, with same dimensions as
        y_true.
    train_idxs : list of arrays
        The training indices for each fold.
    test_idxs : list of arrays
        The test indices for each fold.
    threshold : float
        The threshold by which to convert predicted probabilities to labels.

    Returns
    -------
    accuracy_by_chances : np.ndarray
        An array with the accuracy by chance on each fold (first dimension)
        and label (second dimension).
    """
    n_labels = len(y_true)
    # Calculate accuracy by chance over each label
    accuracy_by_chances = np.array(
        [cv_accuracy_by_chance(y_true[label],
                               y_pred[label],
                               train_idxs,
                               test_idxs,
                               threshold=threshold)
         for label in range(n_labels)]
    )
    return accuracy_by_chances.T

def cv_multilabel_log_odds_difference(y_true, y_pred, train_idxs, test_idxs, threshold=0.5):
    """Calculates the log-odds difference over CV folds and labels.

    Parameters
    ----------
    y_true : list of array-like
        The true labels, as a list of arrays. The list is over labels, and the
        array is over all samples.
    y_pred : array-like
        The predicted probabilities of each label, with same dimensions as
        y_true.
    train_idxs : list of arrays
        The training indices for each fold.
    test_idxs : list of arrays
        The test indices for each fold.
    threshold : float
        The threshold by which to convert predicted probabilities to labels.

    Returns
    -------
    log_odds_differences : np.ndarray
        An array with the log-odds difference on each fold (first dimension)
        and label (second dimension).
    """
    n_labels = len(y_true)
    # Calculate log-odds difference over each label
    log_odds_differences = np.array(
        [cv_log_odds_difference(y_true[label],
                                y_pred[label],
                                train_idxs,
                                test_idxs,
                                threshold=threshold)
         for label in range(n_labels)]
    )
    return log_odds_differences


def cv_metric(y_true, y_pred, test_idxs, metric, **kwargs):
    """Calculates a metric over CV folds.

    Parameters
    ----------
    y_true : array-like
        The true labels across all samples.
    y_pred : array-like
        The predicted probabilities of each label, with same dimensions as
        y_true.
    test_idxs : list of arrays
        The test indices for each fold.
    metric : function
        The metric to use.

    Returns
    -------
    metrics : np.ndarray
        An array with the metric on each fold.
    """
    n_folds = len(test_idxs)
    metrics = np.zeros(n_folds)

    for idx, test_idx in enumerate(test_idxs):
        # Obtain test samples
        y_true_test = y_true[test_idx]
        y_pred_test = y_pred[test_idx]
        # Calculate metric
        metrics[idx] = metric(y_true_test, y_pred_test, **kwargs)
    return metrics


def cv_multilabel_metric(y_true, y_pred, test_idxs, metric='roc_auc'):
    """Calculates the log-odds difference over CV folds and labels.

    Parameters
    ----------
    y_true : list of array-like
        The true labels, as a list of arrays. The list is over labels, and the
        array is over all samples.
    y_pred : array-like
        The predicted probabilities of each label, with same dimensions as
        y_true.
    test_idxs : list of arrays
        The test indices for each fold.
    metric : string
        The metric to use. Options include 'roc_auc' or 'pr_auc'.

    Returns
    -------
    metrics : np.ndarray
        An array with the metric on each fold (first dimension) and label
        (second dimension).
    """
    if metric == 'roc_auc':
        metric_fn = roc_auc_score
    elif metric == 'pr_auc':
        metric_fn = average_precision_score
    else:
        raise ValueError(f'Metric "{metric}" not available.')

    n_labels = len(y_true)
    metrics = np.array(
        [cv_metric(y_true[label], y_pred[label], test_idxs, metric_fn)
         for label in range(n_labels)]
    )
    return metrics


def preprocess(text):
    """Normalize some distracting parts of text data.
    URLS, phone numbers and email addresses are remove to protect people's
    identities if that content ends up in our data. Accents are removed and
    everything is case-folded to reduce the character and type vocab that we
    deal with downstream.
    Parameters
    ----------
    text : str
    Returns
    -------
    str
    """
    text = ftfy.fix_text(text)
    text = text.lower()
    text = pp.normalize.whitespace(text)
    text = text.replace('\n', ' ')
    text = pp.replace.urls(text, repl='URL')
    text = pp.replace.phone_numbers(text, repl='PHONE')
    text = pp.replace.emails(text, repl='EMAIL')
    text = pp.remove.accents(text)
    return text


def cv_wrapper(x, y, model_builder, model_kwargs={}, compile_kwargs={},
               batch_size=32, max_epochs=50, n_folds=5, val_frac=0.2,
               lr=0.001, refit=False, refit_fold=False,
               unwrap_predictions=False, callbacks=None, sample_weights=None,
               random_state=None, store_models=False, verbose=False,
               cv_verbose=False):
    """Cross-validate a model and evaluate its predictive accuracy.

    Within each fold, the model is fit first with early stopping on a
    validation set, specified by `val_frac`, to determine how many epochs to
    run. The model is then reset and run a second time with no early stopping
    or validation set for that many epochs.

    Parameters
    ----------
    x : list
        Model inputs, with each list entry denoting a sample.
    y : list
        Model outputs, with each list entry denoting a sample.
    model_builder : function
        A function that builds a model in the Functional API. This is used to
        initialize fresh models at the beginning of each cross-validation fold.
    model_kwargs : dict
        Keyword arguments for model builder.
    compile_kwargs : dict
        Keyword arguments for compiling the model. If empty dictionary, the
        model is compiled with categorical cross-entropy and an Adam optimizer.
    batch_size : int
        Batch size.
    max_epochs : int
        The maximum number of epochs to run.
    n_folds : int
        The number of cross-validation folds.
    val_frac : float
        The percentage of training data to use for early stopping in the first
        model fit.
    lr : float
        The learning rate.
    refit : bool
        If True, refits the model to the entire dataset and returns that
        history and trained model. If False, only trains across the folds,
        and returns the models and histories for each fold.
    refit_fold : bool
        If True, refits the model within the fold to the entire training set
        using the early stopping number of epochs.
    unwrap_predictions : bool
        If True, rearranges the test predictions to have outer list correspond
        to output, while each inner array corresponds to the predictions for
        that output across all samples, sorted by the input sample index.
    sample_weights : array-like of None
        The weight for each sample during training. If None, assumes all
        samples have the same weight.
    random_state : int, RandomState, or None
        Random state for cross-validation fitting.
    verbose : bool
        Verbosity flag during training.
    cv_verbose : bool
        Verbosity flag over CV folds.

    Returns
    -------
    model : tf.keras.model
        The trained model. If refit is False, a list of trained models across
        folds.
    history : tf.keras.callbacks.History
        The history of each training process. If refit is False, a list of
        histories for each fold.
    train_idxs : list of np.ndarrays
        The training indices per fold.
    test_idxs : list of np.ndarrays
        The test indices per fold.
    test_predictions : list
        The predictions of the trained classifier on the held-out data. If
        unwrap_predictions is True, this is a list corresponding to each output.
        If unwrap_predictions is False, this is a list corresponding to each
        fold.
    test_scores : np.ndarray
        The test scores for each fold.
    model_refit : tf.keras.model (optional)
        The model refit to the entire training set. Only returned if
        refit=True.
    history_refit : tf.keras.callbacks.History (optional)
        The history of the model refit to entire training set. Only returned if
        refit=True.
    """
    # Use the first input to extract the sample
    n_samples = x[0].shape[0]
    n_outputs = len(y)
    if sample_weights is None:
        sample_weights = np.ones(n_samples)
    # Create cross-validation object
    cv = KFold(n_splits=n_folds)
    # Lists for storing information across folds
    n_epochs = np.zeros(n_folds, dtype=int)
    histories = []
    models = []
    train_idxs = []
    test_idxs = []
    test_predictions = []
    test_scores = []
    # Default compiler arguments
    if compile_kwargs == {}:
        compile_kwargs['loss'] = 'binary_crossentropy'
        compile_kwargs['optimizer'] = Adam(0.01)
        compile_kwargs['metrics'] = ['acc']

    if cv_verbose:
        print("Beginning cross-validation...")

    # Iterate over group folds
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(x[0])):
        if cv_verbose:
            print(f"Fold {fold_idx + 1} / {n_folds}")
        # Store training and testing indices
        train_idxs.append(train_idx)
        test_idxs.append(test_idx)
        # Create training set for current fold
        x_train = [x_input[train_idx] for x_input in x]
        y_train = [y_output[train_idx] for y_output in y]
        sample_weights_train = sample_weights[train_idx]
        # Generate validation set, while respecting group structure
        cv_val = ShuffleSplit(n_splits=1,
                              test_size=val_frac,
                              random_state=random_state)
        # Obtain indices for single validation set by applying generator
        train_val_idx, test_val_idx = next(cv_val.split(x_train[0]))
        x_train_val = [x_input[train_val_idx] for x_input in x_train]
        y_train_val = [y_output[train_val_idx] for y_output in y_train]
        x_test_val = [x_input[test_val_idx] for x_input in x_train]
        y_test_val = [y_output[test_val_idx] for y_output in y_train]
        sample_weights_val = sample_weights_train[train_val_idx]
        if cv_verbose:
            print("Fitting model to training set...")
        # Compile model and fit to training data
        tf.keras.backend.clear_session()
        model = model_builder(**model_kwargs)
        model.compile(**compile_kwargs)
        history = model.fit(x=x_train_val,
                            y=y_train_val,
                            batch_size=batch_size,
                            epochs=max_epochs,
                            validation_data=(x_test_val, y_test_val),
                            verbose=verbose,
                            callbacks=callbacks,
                            sample_weight=sample_weights_val)
        # Get number of epochs
        val_loss = history.history['val_loss']
        n_epochs_best = int(1 + np.argmin(val_loss))
        n_epochs[fold_idx] = n_epochs_best
        if refit_fold:
            del model
            del history
            gc.collect()
            if cv_verbose:
                print(f"Refitting to entire training set with {n_epochs[fold_idx]} epochs.")
            tf.keras.backend.clear_session()
            model = model_builder(**model_kwargs)
            model.compile(**compile_kwargs)
            history = model.fit(x=x_train,
                                y=y_train,
                                batch_size=batch_size,
                                epochs=n_epochs_best,
                                verbose=verbose,
                                sample_weight=sample_weights_train)
        if store_models:
            models.append(model)
            histories.append(history)
        if cv_verbose:
            print("Fitting complete. Running test predictions...")

        # Evaluate the model on the test set
        x_test = [x_input[test_idx] for x_input in x]
        y_test = [y_output[test_idx] for y_output in y]
        # Predictions on the test set
        test_predictions.append(model.predict(x_test))
        # Evaluation on the test set
        test_scores.append(model.evaluate(x=x_test, y=y_test))
        # Save some memory
        del model
        del history
        gc.collect()
        if cv_verbose:
            print("Fold complete.")

    if cv_verbose:
        print("Cross-validation complete.")

    # Convert test scores to numpy array
    test_scores = np.array(test_scores)
    # Convert test predictions into one array, sorted by sample, per output
    if unwrap_predictions:
        sorted_idxs = np.argsort(np.concatenate(test_idxs))
        test_predictions = [
            np.concatenate([test_predictions[fold][output]
                            for fold in range(n_folds)], axis=0)[sorted_idxs]
            for output in range(n_outputs)]
    # Begin compiling results
    results = {
        'n_epochs': n_epochs,
        'models': models,
        'histories': histories,
        'train_idxs': train_idxs,
        'test_idxs': test_idxs,
        'test_predictions': test_predictions,
        'test_scores': test_scores}
    # Refit model to entire dataset if needed
    if refit:
        n_epochs_refit = int(np.round(np.mean(n_epochs)))
        if cv_verbose:
            print(f"Refitting model using {n_epochs_refit} epochs.")
        tf.keras.backend.clear_session()
        model_refit = model_builder(**model_kwargs)
        model_refit.compile(**compile_kwargs)
        history_refit = model_refit.fit(x=x,
                                        y=y,
                                        batch_size=batch_size,
                                        epochs=n_epochs_refit,
                                        verbose=verbose,
                                        sample_weight=sample_weights)
        if cv_verbose:
            print("Model fit complete.")
        results['model_refit'] = model_refit
        results['history_refit'] = history_refit
        del model_refit
        del history_refit
        gc.collect()

    return results


def cv_wrapper_memory_friendly(
    x, y, model_builder, model_kwargs={}, compile_kwargs={},
    batch_size=32, max_epochs=50, n_folds=5, val_frac=0.2, refit=False,
    unwrap_predictions=False, callbacks=None, sample_weights=None,
    random_state=None, cv_verbose=False, save=None):
    """Cross-validate a model and evaluate its predictive accuracy.

    Within each fold, the model is fit first with early stopping on a
    validation set, specified by `val_frac`, to determine how many epochs to
    run. The model is then reset and run a second time with no early stopping
    or validation set for that many epochs.

    Parameters
    ----------
    x : list
        Model inputs, with each list entry denoting a sample.
    y : list
        Model outputs, with each list entry denoting a sample.
    model_builder : function
        A function that builds a model in the Functional API. This is used to
        initialize fresh models at the beginning of each cross-validation fold.
    model_kwargs : dict
        Keyword arguments for model builder.
    compile_kwargs : dict
        Keyword arguments for compiling the model. If empty dictionary, the
        model is compiled with categorical cross-entropy and an Adam optimizer.
    batch_size : int
        Batch size.
    max_epochs : int
        The maximum number of epochs to run.
    n_folds : int
        The number of cross-validation folds.
    val_frac : float
        The percentage of training data to use for early stopping in the first
        model fit.
    lr : float
        The learning rate.
    refit : bool
        If True, refits the model to the entire dataset and returns that
        history and trained model. If False, only trains across the folds,
        and returns the models and histories for each fold.
    refit_fold : bool
        If True, refits the model within the fold to the entire training set
        using the early stopping number of epochs.
    unwrap_predictions : bool
        If True, rearranges the test predictions to have outer list correspond
        to output, while each inner array corresponds to the predictions for
        that output across all samples, sorted by the input sample index.
    sample_weights : array-like of None
        The weight for each sample during training. If None, assumes all
        samples have the same weight.
    random_state : int, RandomState, or None
        Random state for cross-validation fitting.
    verbose : bool
        Verbosity flag during training.
    cv_verbose : bool
        Verbosity flag over CV folds.

    Returns
    -------
    model : tf.keras.model
        The trained model. If refit is False, a list of trained models across
        folds.
    history : tf.keras.callbacks.History
        The history of each training process. If refit is False, a list of
        histories for each fold.
    train_idxs : list of np.ndarrays
        The training indices per fold.
    test_idxs : list of np.ndarrays
        The test indices per fold.
    test_predictions : list
        The predictions of the trained classifier on the held-out data. If
        unwrap_predictions is True, this is a list corresponding to each output.
        If unwrap_predictions is False, this is a list corresponding to each
        fold.
    test_scores : np.ndarray
        The test scores for each fold.
    model_refit : tf.keras.model (optional)
        The model refit to the entire training set. Only returned if
        refit=True.
    history_refit : tf.keras.callbacks.History (optional)
        The history of the model refit to entire training set. Only returned if
        refit=True.
    """
    # Use the first input to extract the sample
    n_samples = x[0].shape[0]
    n_outputs = len(y)
    if sample_weights is None:
        sample_weights = np.ones(n_samples)
    # Create cross-validation object
    cv = KFold(n_splits=n_folds)
    # Lists for storing information across folds
    n_epochs = np.zeros(n_folds, dtype=int)
    train_idxs = []
    test_idxs = []
    test_predictions = []
    test_scores = np.zeros((n_folds, n_outputs + 1))
    # Default compiler arguments
    if compile_kwargs == {}:
        compile_kwargs['loss'] = 'binary_crossentropy'
        compile_kwargs['optimizer'] = Adam(0.01)
        compile_kwargs['metrics'] = ['acc']

    if cv_verbose:
        print("Beginning cross-validation...")

    # Iterate over group folds
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(x[0])):
        if cv_verbose:
            print(f"Fold {fold_idx + 1} / {n_folds}")
        # Store training and testing indices
        train_idxs.append(train_idx)
        test_idxs.append(test_idx)
        # Create training set for current fold
        x_train = [x_input[train_idx] for x_input in x]
        y_train = [y_output[train_idx] for y_output in y]
        sample_weights_train = sample_weights[train_idx]
        # Generate validation set, while respecting group structure
        cv_val = ShuffleSplit(n_splits=1,
                              test_size=val_frac,
                              random_state=random_state)
        # Obtain indices for single validation set by applying generator
        train_val_idx, test_val_idx = next(cv_val.split(x_train[0]))
        x_train_val = [x_input[train_val_idx] for x_input in x_train]
        y_train_val = [y_output[train_val_idx] for y_output in y_train]
        x_test_val = [x_input[test_val_idx] for x_input in x_train]
        y_test_val = [y_output[test_val_idx] for y_output in y_train]
        sample_weights_val = sample_weights_train[train_val_idx]
        if cv_verbose:
            print("Fitting model to training set...")
        # Compile model and fit to training data
        queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=memory_friendly_helper,
            args=(
                model_builder,
                x_train_val,
                y_train_val,
                batch_size,
                max_epochs,
                callbacks,
                sample_weights_val,
                model_kwargs,
                compile_kwargs,
                queue,
                x_test_val,
                y_test_val,
                None,
                None,
                True,
                None))
        process.start()
        process.join()
        if cv_verbose:
            print('Process exited. Beginning full fit.')
        results = queue.get()
        n_epochs[fold_idx] = results['n_epochs']

        # Evaluate the model on the test set
        x_test = [x_input[test_idx] for x_input in x]
        y_test = [y_output[test_idx] for y_output in y]

        if cv_verbose:
            print(f"Refitting to entire training set with {n_epochs[fold_idx]} epochs.")
        queue = ArrayQueue()
        process = multiprocessing.Process(
            target=memory_friendly_helper,
            args=(
                model_builder,
                x_train,
                y_train,
                batch_size,
                n_epochs[fold_idx],
                None,
                sample_weights_train,
                model_kwargs,
                compile_kwargs,
                queue,
                None,
                None,
                x_test,
                y_test,
                True,
                None))
        process.start()
        process.join()
        if cv_verbose:
            print('Process exited. Storing predictions.')
        predictions = queue.get()
        # Predictions on the test set
        test_predictions.append(predictions)
        # Evaluation on the test set
        test_scores[fold_idx] = queue.get()

        if cv_verbose:
            print("Fold complete.")

    if cv_verbose:
        print("Cross-validation complete.")

    # Convert test predictions into one array, sorted by sample, per output
    if unwrap_predictions:
        sorted_idxs = np.argsort(np.concatenate(test_idxs))
        test_predictions = [
            np.concatenate([test_predictions[fold][output]
                            for fold in range(n_folds)], axis=0)[sorted_idxs]
            for output in range(n_outputs)]
    # Begin compiling results
    results = {
        'n_epochs': n_epochs,
        'train_idxs': train_idxs,
        'test_idxs': test_idxs,
        'test_predictions': test_predictions,
        'test_scores': test_scores}
    # Refit model to entire dataset if needed
    if refit:
        n_epochs_refit = int(np.round(np.mean(n_epochs)))
        if cv_verbose:
            print(f"Refitting model using {n_epochs_refit} epochs.")
        process = multiprocessing.Process(
            target=memory_friendly_helper,
            args=(
                model_builder,
                x,
                y,
                batch_size,
                n_epochs_refit,
                None,
                sample_weights,
                model_kwargs,
                compile_kwargs,
                None,
                None,
                None,
                None,
                None,
                True,
                save))
        process.start()
        process.join()
    return results


def memory_friendly_helper(
    model_builder, x_train, y_train, batch_size, max_epochs,
    callbacks, sample_weights, model_kwargs, compile_kwargs,
    queue, x_val=None, y_val=None, x_test=None, y_test=None, verbose=True,
    save=None
):
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    gpu = gpus[0]
    tf.config.experimental.set_visible_devices(gpu, 'GPU')
    tf.config.experimental.set_memory_growth(gpu, True)

    # Compile model and fit to training data
    model = model_builder(**model_kwargs)
    model.compile(**compile_kwargs)

    if x_val is not None:
        history = model.fit(x=x_train,
                            y=y_train,
                            batch_size=batch_size,
                            epochs=max_epochs,
                            validation_data=(x_val, y_val),
                            verbose=verbose,
                            callbacks=callbacks,
                            sample_weight=sample_weights)
        # Get number of epochs
        val_loss = history.history['val_loss']
        results = {}
        results['n_epochs'] = int(1 + np.argmin(val_loss))
        queue.put(results)
    elif x_test is not None:
        history = model.fit(x=x_train,
                            y=y_train,
                            batch_size=batch_size,
                            epochs=max_epochs,
                            verbose=verbose,
                            sample_weight=sample_weights)

        print('Running test predictions and scores.')
        # Predictions on the test set
        predictions = np.array(model.predict(x_test)).squeeze()
        # Evaluation on the test set
        scores = np.array(model.evaluate(x=x_test, y=y_test))

        print('Predictions placed inside results dict.')
        queue.put(predictions)
        queue.put(scores)
    else:
        history = model.fit(x=x_train,
                            y=y_train,
                            batch_size=batch_size,
                            epochs=max_epochs,
                            verbose=verbose,
                            sample_weight=sample_weights)
        print('Saving weights.')
        model.save_weights(save)
    print('Exiting process.')