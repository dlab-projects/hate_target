import numpy as np
import pandas as pd

from . import keys
from .utils import preprocess


def load_gab_hate_corpus(path):
    """Loads the GAB Hate Corpus.
    
    Parameters
    ----------
    path : string
        The path to the GAB Hate Corpus.
        
    Returns
    -------
    gab : pd.DataFrame
        A dataframe containing the GAB Hate Corpus.
    is_target : pd.DataFrame
        A dataframe identifying unique comments, and what they target according
        to the annotator.
    """
    gab = pd.read_csv(path, delimiter='\t')
    # Assign either ideology or politics to a common label
    gab['IDL_POL'] = ((gab['IDL'] == 1) | (gab['POL'] == 1)).astype('int')
    # Rename columns    
    gab = gab.rename({
        'ID': 'comment_id',
        'Annotator': 'labeler_id',
        'Text': 'text',
        'REL': 'target_religion',
        'RAE': 'target_race',
        'SXO': 'target_sexuality',
        'GEN': 'target_gender',
        'NAT': 'target_origin',
        'IDL_POL': 'target_politics',
        'MPH': 'target_disability'}, axis=1)
    # Obtain target columns
    target_cols = sorted([col for col in gab.columns if 'target' in col])
    # Shrink dataset down
    gab = gab[['comment_id', 'labeler_id', 'text'] + target_cols]
    # Obtain unique comments
    comments = gab[['comment_id', 'text']].drop_duplicates().sort_values('comment_id')
    # Obtain valid annotations
    annotations = gab[~gab[target_cols].isna().any(axis=1)]
    agreement = annotations[['comment_id'] + target_cols].groupby('comment_id').agg('mean')
    is_target = (agreement >= 0.5).astype('int'
        ).reset_index(level=0
        ).merge(right=comments, how='left')
    is_target = is_target[['comment_id', 'text'] + target_cols]
    # Preprocess text
    is_target['predict_text'] = is_target['text'].apply(lambda x: preprocess(x))
    return gab, is_target


def load_subgroup(data_path, group, threshold=0.5, text_col='predict_text'):
    """Load a subgroup from the Measuring Hate Speech dataset.
    
    Parameters
    ----------
    data_path : string
        The path to the Measuring Hate Speech dataset.
    group : string
        The subgroup.
    threshold : float
        The threshold for annotator agreement.
    text_col : string
        The column to use for the input text.
    
    Returns
    -------
    data : pd.DataFrame
        The subgroup data.
    x : list
        The input data (text).
    y_hard : list
        The labels, thresholded.
    y_soft : list
        The soft labels (fraction of annotators saying the comment targets the
        group).
    """
    data = pd.read_feather(data_path)
    comments = data[['comment_id', text_col]].drop_duplicates().sort_values('comment_id')

    if group == 'race':
        target_cols = sorted(keys.target_race_cols)

    # Get targets
    targets = data[['comment_id'] + target_cols]
    # Calculate fraction of annotators agreement on target
    agreement = targets.groupby('comment_id'
        ).agg('mean'
        ).reset_index(level=0)
    # Make hard label
    hard = (agreement >= threshold).astype('int'
        ).reset_index(level=0
        ).merge(right=comments, how='left')
    agreement = agreement.merge(right=comments, how='left') 
    # Input features
    x = agreement[text_col].values
    # Assign labels (hard or soft labels)
    y_soft = [agreement[col].values[..., np.newaxis] for col in target_cols]
    y_hard = [hard[col].values[..., np.newaxis] for col in target_cols]
    return data, x, y_hard, y_soft