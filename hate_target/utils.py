import numpy as np
import pickle

from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score
from tensorflow.keras.optimizers import Adam


def analyze_experiment(path, n_groups=8):
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
        _, y, _, _, _, test_predictions, test_scores, chance = \
            pickle.load(file)

    # Test loss
    test_loss = test_scores[:, 0]
    # Test loss on labels
    test_loss_labels = test_scores[:, 1:(n_groups + 1)]
    # Test accuracies
    test_accs = test_scores[:, (n_groups + 1):]
    # Accuracy over chance
    acc_over_chance = test_accs / chance
    # Log-Odds Difference
    log_odds_diff = np.log(test_accs / (1 - test_accs)) - np.log(chance / (1 - chance))
    # AUC ROC
    roc_aucs = np.array(
        [roc_auc_score(y[i].ravel(), test_predictions[i].ravel())
         for i in range(n_groups)])
    pr_aucs = auc_prcs = np.array(
        [average_precision_score(y[i].ravel(), test_predictions[i].ravel())
         for i in range(n_groups)])

    results = {
        'test_loss': test_loss,
        'test_loss_labels': test_loss_labels,
        'test_accs': test_accs,
        'acc_over_chance': acc_over_chance,
        'log_odds_difference': log_odds_diff,
        'roc_aucs': roc_aucs,
        'pr_aucs': pr_aucs}
    return results


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
        y_true_train = y_true[train_idx]
        y_true_test = y_true[test_idx]
        y_pred_test = (y_pred[test_idx] >= threshold).astype('int')
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
        y_true_train = y_true[train_idx]
        y_true_test = y_true[test_idx]
        y_pred_test = (y_pred[test_idx] >= threshold).astype('int')
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


def cv_wrapper(x, y, model_builder, model_kwargs={}, compile_kwargs={},
               batch_size=32, max_epochs=50, n_folds=5, val_frac=0.2,
               lr=0.001, refit=False, refit_fold=False,
               unwrap_predictions=False, report_chance=False,
               callbacks=None, sample_weights=None, random_state=None,
               verbose=False, cv_verbose=False):
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
    report_chance : bool
        If True, returns an array that contains the accuracy achieved by a naive
        classifier for each training set.
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
    if report_chance:
        chance = np.zeros((n_folds, len(y)))
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
            if cv_verbose:
                print(f"Refitting to entire training set with {n_epochs[fold_idx]} epochs.")
            model = model_builder(**model_kwargs)
            model.compile(**compile_kwargs)
            history = model.fit(x=x_train,
                                y=y_train,
                                batch_size=batch_size,
                                epochs=n_epochs_best,
                                verbose=verbose,
                                sample_weight=sample_weights_train)
        histories.append(history)
        models.append(model)
        if cv_verbose:
            print("Fitting complete. Running test predictions...")
        # Evaluate the model on the test set
        x_test = [x_input[test_idx] for x_input in x]
        y_test = [y_output[test_idx] for y_output in y]
        # Predictions on the test set
        test_predictions.append(model.predict(x_test))
        # Evaluation on the test set
        test_scores.append(model.evaluate(x=x_test, y=y_test))
        if report_chance:
            chance[fold_idx] = [
                accuracy_by_chance(train.mean(), test.mean())
                for train, test in zip(y_train, y_test)]
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
    results = (n_epochs, models, histories, train_idxs, test_idxs,
               test_predictions, test_scores)
    # Refit model to entire dataset if needed
    if refit:
        n_epochs_refit = int(np.round(np.mean(n_epochs)))
        if cv_verbose:
            print(f"Refitting model using {n_epochs_refit} epochs.")
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
        results = results + (model_refit, history_refit)
    if report_chance:
        results = results + (chance,)
    return results
