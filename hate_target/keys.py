# Hate speech construct items
items = [
    "sentiment",
    "respect",
    "insult",
    "humiliate",
    "status",
    "dehumanize",
    "violence",
    "genocide",
    "attack_defend",
    "hatespeech"]

"""
Target identities
"""
# Column names for target groups
target_groups = [
    'target_race',
    'target_religion',
    'target_origin',
    'target_gender',
    'target_sexuality',
    'target_age',
    'target_disability',
    'target_politics']

target_labels = [group.split('_')[1].capitalize()
                 for group in target_groups]

# Targets race columns
target_race_cols = [
    'target_race_asian',
    'target_race_black',
    'target_race_latinx',
    'target_race_middle_eastern',
    'target_race_native_american',
    'target_race_pacific_islander',
    'target_race_white',
    'target_race_other']

# Targets religion columns
target_religion_cols = [
    'target_religion_atheist',
    'target_religion_buddhist',
    'target_religion_christian',
    'target_religion_hindu',
    'target_religion_jewish',
    'target_religion_mormon',
    'target_religion_muslim',
    'target_religion_other',]

# Targets national origin columns
target_origin_cols = [
    'target_origin_immigrant',
    'target_origin_migrant_worker',
    'target_origin_specific_country',
    'target_origin_undocumented',
    'target_origin_other']

# Targets gender column
target_gender_cols = [
    'target_gender_men',
    'target_gender_non_binary',
    'target_gender_transgender_men',
    'target_gender_transgender_unspecified',
    'target_gender_transgender_women',
    'target_gender_women',
    'target_gender_other']

# Targets sexuality column
target_sexuality_cols = [
    'target_sexuality_bisexual',
    'target_sexuality_gay',
    'target_sexuality_lesbian',
    'target_sexuality_straight',
    'target_sexuality_other']

# Targets age column
target_age_cols = [
    'target_age_children',
    'target_age_teenagers',
    'target_age_young_adults',
    'target_age_middle_aged',
    'target_age_seniors',
    'target_age_other']

# Targets disability column
target_disability_cols = [
    'target_disability_physical',
    'target_disability_cognitive',
    'target_disability_neurological',
    'target_disability_visually_impaired',
    'target_disability_hearing_impaired',
    'target_disability_unspecific',
    'target_disability_other']

# Targets politics column
target_politics_cols = [
    'target_politics_alt_right',
    'target_politics_communist',
    'target_politics_conservative',
    'target_politics_democrat',
    'target_politics_green_party',
    'target_politics_leftist',
    'target_politics_liberal',
    'target_politics_libertarian',
    'target_politics_republican',
    'target_politics_socialist',
    'target_politics_other']

# All targets
target_cols = target_race_cols + \
              target_religion_cols + \
              target_origin_cols + \
              target_gender_cols + \
              target_sexuality_cols + \
              target_age_cols + \
              target_disability_cols + \
              target_politics_cols
