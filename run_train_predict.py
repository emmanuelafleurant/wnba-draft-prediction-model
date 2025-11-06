import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

ROOT = os.path.dirname(__file__)
TRAIN_CSV = os.path.join(ROOT, 'wnbadraft.csv')
TEST_CSV_CANDIDATES = [
    os.path.join(ROOT, 'ML 2025 WNBA Data.csv'),
    os.path.join(ROOT, 'ML_2025_WNBA_Data.csv'),
    os.path.join(ROOT, 'ml_2025_wnba_data.csv')
]
OUT_PRED = os.path.join(ROOT, 'predicted_first_round_2025.csv')

print('Loading training CSV:', TRAIN_CSV)
df = pd.read_csv(TRAIN_CSV)
print('Columns:', list(df.columns))

# helper to find a column name flexibly
def find_col(possible):
    for p in possible:
        if p in df.columns:
            return p
    return None

overall_col = find_col(['overall_pick', 'Overall', 'overall', 'overall pick', 'overall_pick '])
year_col = find_col(['year', 'Year', 'draft_year', 'draft_year '])
college_col = find_col(['college', 'College', 'college/former', 'college/former '])
position_col = find_col(['position', 'Position', 'pos'])
height_col = find_col(['height', 'Height', 'height_in'])

if overall_col is None or year_col is None:
    print('Cannot find overall pick or year columns. Found:', overall_col, year_col)
    sys.exit(1)

print('Using columns:', 'overall=', overall_col, 'year=', year_col, 'college=', college_col, 'position=', position_col, 'height=', height_col)

# Build target: top12
# If overall_pick is NaN for undrafted, we'll treat as not top12

df[overall_col] = pd.to_numeric(df[overall_col], errors='coerce')
df[year_col] = pd.to_numeric(df[year_col], errors='coerce')
df['top12'] = df[overall_col].apply(lambda x: 1 if pd.notnull(x) and x <= 12 else 0)

# restrict to rows with a year and non-null player
if 'player' in df.columns:
    df = df[pd.notnull(df['player'])]

# Convert college to string
if college_col:
    df[college_col] = df[college_col].fillna('')
else:
    print('No college column found; creating empty college')
    df['college'] = ''
    college_col = 'college'

# Height parsing

def parse_height(x):
    if pd.isnull(x):
        return np.nan
    if isinstance(x, (int, float)):
        return x
    s = str(x).strip()
    # formats like 6-2, 6'2, 6 2
    for sep in ["-", "'", " "]:
        if sep in s:
            parts = s.split(sep)
            if len(parts) >= 2:
                try:
                    feet = float(parts[0])
                    inches = float(parts[1])
                    return feet * 12 + inches
                except:
                    break
    # decimal feet like 6.25
    try:
        f = float(s)
        # if reasonable feet
        if 4 <= f <= 8:
            return f * 12
    except:
        pass
    return np.nan

if height_col:
    df['height_in'] = df[height_col].apply(parse_height)
else:
    df['height_in'] = np.nan

# Position dummies
if position_col:
    pos_dummies = pd.get_dummies(df[position_col].fillna('UNK'), prefix='pos')
else:
    pos_dummies = pd.DataFrame(index=df.index)

# K-fold out-of-fold target encoding for college
kf = KFold(n_splits=5, shuffle=True, random_state=42)
college_te = pd.Series(index=df.index, dtype=float)
for train_idx, val_idx in kf.split(df):
    means = df.iloc[train_idx].groupby(college_col)['top12'].mean()
    college_te.iloc[val_idx] = df.iloc[val_idx][college_col].map(means)
# fill unknowns with global mean
college_te.fillna(df['top12'].mean(), inplace=True)
df['college_te'] = college_te
# college frequency
college_counts = df[college_col].map(df[college_col].value_counts())
# normalize counts
df['college_freq'] = college_counts.fillna(0)

# assemble features
feature_cols = ['height_in', 'college_te', 'college_freq']
# add position dummies
for c in pos_dummies.columns:
    df[c] = pos_dummies[c]
    feature_cols.append(c)

# interaction: height * pos
for c in pos_dummies.columns:
    inter = f'height_x_{c}'
    df[inter] = df['height_in'] * df[c]
    feature_cols.append(inter)

# Drop rows without year or with year >=2025 in training
train_df = df[df[year_col] < 2025]
# For training, drop rows where overall_pick is NaN? We'll keep all but only rows with a known top12 label
train_df = train_df[pd.notnull(train_df[year_col])]
# drop rows with missing features
train_df = train_df.dropna(subset=['top12'])

print('Training rows:', len(train_df))
X = train_df[feature_cols].fillna(0)
y = train_df['top12']

# Train/test split for quick validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)
print('Validation score (accuracy):', model.score(X_val, y_val))

# permutation importance on validation set
print('Computing permutation importances...')
perm = permutation_importance(model, X_val, y_val, n_repeats=10, random_state=42, n_jobs=-1)
imp_df = pd.DataFrame({'feature': feature_cols, 'importance_mean': perm.importances_mean, 'importance_std': perm.importances_std})
imp_df = imp_df.sort_values('importance_mean', ascending=False)
print(imp_df)
imp_df.to_csv(os.path.join(ROOT, 'pre_draft_permutation_importances_simple.csv'), index=False)

test_path = None
for p in TEST_CSV_CANDIDATES:
    if os.path.exists(p):
        test_path = p
        break
if test_path is None:
    print('No 2025 candidate file found in expected names. Looking for rows in train DF with year==2025')
    test_df = df[df[year_col] == 2025].copy()
else:
    print('Loading 2025 candidates from', test_path)
    # the 2025 csv may have a few descriptive lines before the true header
    # try to detect the header row by searching for a line that contains 'Player' or 'Player Name'
    header_row = None
    with open(test_path, 'r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            if 'Player' in line or 'Player Name' in line:
                header_row = i
                break
    if header_row is not None:
        # read using detected header (pandas uses zero-indexed skiprows -> header at row header_row)
        test_df = pd.read_csv(test_path, header=header_row)
    else:
        test_df = pd.read_csv(test_path)
    # ensure columns exist
    # map some common 2025 column names to the ones we expect
    col_map = {}
    if 'Player Name' in test_df.columns:
        col_map['Player Name'] = 'player'
    if 'College/Country' in test_df.columns:
        col_map['College/Country'] = college_col
    if 'Height' in test_df.columns:
        col_map['Height'] = height_col if height_col else 'height_in'
    if 'Position' in test_df.columns:
        col_map['Position'] = position_col if position_col else 'position'
    if col_map:
        test_df = test_df.rename(columns=col_map)

    if college_col in test_df.columns:
        test_df[college_col] = test_df[college_col].fillna('')
    else:
        test_df[college_col] = ''
    if height_col in test_df.columns:
        test_df['height_in'] = test_df[height_col].apply(parse_height)
    else:
        test_df['height_in'] = np.nan
    if position_col in test_df.columns:
        for c in pos_dummies.columns:
            test_df[c] = 0
        test_pos = pd.get_dummies(test_df[position_col].fillna('UNK'), prefix='pos')
        for c in test_pos.columns:
            if c in test_df.columns:
                test_df[c] = test_pos[c]
            else:
                test_df[c] = test_pos[c]

# map college_te using full training data mapping
train_college_means = train_df.groupby(college_col)['top12'].mean()
if college_col in test_df.columns:
    test_df['college_te'] = test_df[college_col].map(train_college_means)
    test_df['college_te'].fillna(train_df['top12'].mean(), inplace=True)
else:
    test_df['college_te'] = train_df['top12'].mean()

# college_freq from training counts
train_college_counts = train_df[college_col].value_counts()
if college_col in test_df.columns:
    test_df['college_freq'] = test_df[college_col].map(train_college_counts).fillna(0)
else:
    test_df['college_freq'] = 0

# ensure feature columns exist in test
for c in feature_cols:
    if c not in test_df.columns:
        test_df[c] = 0

X_test = test_df[feature_cols].fillna(0)
probs = model.predict_proba(X_test)[:, 1]
test_df['pred_prob_top12'] = probs

# Choose top 12 by probability
top12 = test_df.sort_values('pred_prob_top12', ascending=False).head(12)
print('Top 12 predictions for 2025:')
cols_to_show = ['player'] if 'player' in test_df.columns else []
for idx, row in top12.iterrows():
    name = row.get('player', '')
    print(f"{name} - prob={row['pred_prob_top12']:.3f} - college={row.get(college_col, '')} - pos={row.get(position_col, '')}")

# Save CSV
save_cols = ['player', college_col, position_col, 'pred_prob_top12'] if 'player' in test_df.columns else list(top12.columns)
try:
    top12.to_csv(OUT_PRED, index=False, columns=save_cols)
    print('Saved top12 predictions to', OUT_PRED)
except Exception as e:
    print('Could not save CSV:', e)

print('Done')
