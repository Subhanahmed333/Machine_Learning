
import pickle
import pandas as pd

# Load pipeline
pipe = pickle.load(open('pipe.pkl', 'rb'))

test_input = [[82, "Private", "HS-grad", 9, "Widowed", "Exec-managerial", 
               "Not-in-family", "White", "Female", 0, 4356, 18, "United-States"]]

# Convert to DataFrame with proper column names (recommended for pipeline compatibility)
test_df = pd.DataFrame(test_input, columns=[
    'age', 'workclass', 'education', 'education.num', 'marital.status',
    'occupation', 'relationship', 'race', 'sex', 'capital.gain',
    'capital.loss', 'hours.per.week', 'native.country'
])

# Predict
pipe.predict(test_df)
