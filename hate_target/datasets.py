import pandas as pd

from .utils import preprocess


def load_gab_hate_corpus(path):
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