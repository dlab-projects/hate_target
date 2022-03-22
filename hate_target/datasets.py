import numpy as np
import pandas as pd

from . import keys
from .utils import preprocess


def load_gab_hate_corpus(
    path, merge_idl_pol=True, fillna=False, min_annotators=1, threshold=0.5
):
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
    if merge_idl_pol:
        gab['IDL_POL'] = ((gab['IDL'] == 1) | (gab['POL'] == 1)).astype('int')
        pol_col = 'IDL_POL'
    else:
        pol_col = 'POL'

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
        pol_col: 'target_politics',
        'MPH': 'target_disability'}, axis=1)
    # Obtain target columns
    target_cols = sorted([col for col in gab.columns if 'target' in col])
    # Shrink dataset down
    gab = gab[['comment_id', 'labeler_id', 'text'] + target_cols]
    # Obtain unique comments
    comments = gab[['comment_id', 'text']].drop_duplicates().sort_values('comment_id')
    # Obtain valid annotations
    if fillna:
        annotations = gab[['comment_id'] + target_cols].fillna(0).copy()
    else:
        annotations = gab[~gab[target_cols].isna().any(axis=1)].copy()
    # Get comments which have a minimum number of annotations
    comment_counts = annotations['comment_id'].value_counts()
    annotations = annotations[
        annotations['comment_id'].isin(
            comment_counts[comment_counts >= min_annotators].index
        )]
    # Calculate annotator agreement
    agreement = annotations[['comment_id'] + target_cols].groupby('comment_id').agg('mean')
    is_target = (agreement >= threshold).astype('int'
        ).reset_index(level=0
        ).merge(right=comments, how='left')
    is_target = is_target[['comment_id', 'text'] + target_cols]
    # Preprocess text
    is_target['predict_text'] = is_target['text'].apply(lambda x: preprocess(x))
    return gab, is_target


def grab_subgroup_cols(data, subgroup):
    """Loads subgroup columns from the hate speech dataset.
    
    Parameters
    ----------
    data : pd.DataFrame
        The Measuring Hate Speech data.
    subgroup : string
        The subgroup to extract.
        
    Returns
    -------
    targets : pd.DataFrame
        The targets for each comment ID.
    target_cols : list
        The column names for the targets.
    """
    if subgroup == 'race':
        target_cols = sorted(keys.target_race_cols)
        # Get targets
        targets = data[['comment_id'] + target_cols].copy()
    elif subgroup == 'gender':
        target_cols = sorted(keys.target_gender_cols)
        # Get targets
        targets = data[['comment_id'] + target_cols].copy()
        targets['target_gender_transgender'] = targets[[
            'target_gender_transgender_men',
            'target_gender_transgender_women',
            'target_gender_transgender_unspecified']].any(axis=1)
        target_cols = [
            'target_gender_men',
            'target_gender_non_binary',
            'target_gender_transgender',
            'target_gender_women']
        targets = targets[['comment_id'] + target_cols]
    return targets, target_cols


def load_measuring_hate_groups(
    data_path, threshold=0.5, text_col='predict_text',
    comment_id_col='comment_id', rater_quality_path=None,
    target_cols=keys.target_groups
):
    # Load dataset
    data = pd.read_feather(data_path)
    # Performing filtering and rater quality checks
    if rater_quality_path is not None:
        rater_quality = pd.read_csv(rater_quality_path)
        data = filter_annotator_quality(data, rater_quality)
    # Get unique comments
    comments = data[[comment_id_col, text_col]].drop_duplicates().sort_values(comment_id_col)
    # Determine agreement among annotators
    agreement = data[[comment_id_col] + sorted(target_cols)].groupby(comment_id_col).agg('mean')
    is_target = (
        agreement >= threshold
        ).astype('int'
        ).reset_index(level=0
        ).merge(right=comments,
                how='left')
    # Extract data for training models
    x = is_target[text_col].values
    identities = is_target[sorted(target_cols)]
    # Assign labels (hard or soft labels)
    y_soft = [agreement[col].values[..., np.newaxis] for col in identities]
    y_hard = [identities[col].values.astype('int')[..., np.newaxis] for col in identities]
    return data, x, y_hard, y_soft



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

    if group == 'race' or group == 'gender':
        targets, target_cols = grab_subgroup_cols(data, group)
    elif group == 'race_gender':
        targets1, target_cols1 = grab_subgroup_cols(data, 'race')
        targets2, target_cols2 = grab_subgroup_cols(data, 'gender')
        targets = targets1.merge(right=targets2, how='left', on='comment_id')
        target_cols = target_cols1 + target_cols2

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
    return data, x, y_hard, y_soft, target_cols


def filter_missing_items(data, columns=keys.items):
    """Filters the hate speech dataset according to missing data in the survey
    items.
    Parameters
    ----------
    data : pd.DataFrame
        Hate speech dataset.
    Returns
    -------
    data : pd.DataFrame
        Hate speech dataset with rows that contain a missing item removed.
    """
    types = {col: 'int64' for col in columns}
    return data[~data[columns].isna().any(axis=1)].astype(types)


def filter_annotator_quality(
    data, annotator_quality, annotator_col='labeler_id', quality_col='quality_check'
):
    """Filters the hate speech dataset according to rater quality.
    Parameters
    ----------
    data : pd.DataFrame
        Hate speech dataset.
    annotator_quality : pd.DataFrame
        A dataframe containing two columns: one with the annotator ID, and the
        other with a bool indicating whether the annotator is a quality annotator,
        or not.
    annotator_col : string
        The column denoting the annotator ID.
    quality_col : string
        The column denoting the quality flag.
    Returns
    -------
    data : pd.DataFrame
        Hate speech dataset with rows that only contain quality annotators.
    """
    quality_annotators = annotator_quality[annotator_quality[quality_col]][annotator_col].values
    return data[data[annotator_col].isin(quality_annotators)]