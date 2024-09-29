import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def load_data(directory):
    csv_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    
    if not csv_files:
        raise ValueError("No CSV files found in the directory and its subdirectories.")
    
    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True)

def infer_timeline(df):
    possible_assignments = ['Quiz', 'Exam', 'Project', 'Lab', 'Homework', 'Participation']
    assignment_columns = [col for col in df.columns if any(assignment in col for assignment in possible_assignments)]
    
    if not assignment_columns:
        raise ValueError("No assignment columns found in the data.")
    
    filled_columns = {col: df[col].notna().any() for col in assignment_columns}
    latest_filled_column = max(
        (col for col, filled in filled_columns.items() if filled),
        key=lambda col: (col.lower().startswith('quiz'), col.lower().startswith('exam'), col.lower().startswith('project')),
        default=None
    )
    
    if latest_filled_column is None:
        raise ValueError("Could not determine the latest filled assignment.")
    
    timeline = f"After{latest_filled_column.replace(' ', '')}"
    return timeline

def convert_grades_to_numeric(grades):
    grade_dict = {'A+': 4.3, 'A': 4.0, 'A-': 3.7, 'B+': 3.3, 'B': 3.0, 'B-': 2.7,
                  'C+': 2.3, 'C': 2.0, 'C-': 1.7, 'D+': 1.3, 'D': 1.0, 'D-': 0.7, 'F': 0.0}
    return grades.map(grade_dict).fillna(0.0) if isinstance(grades, pd.Series) else grade_dict.get(grades, 0.0)

def convert_numeric_to_grades(numeric_grades):
    grade_dict = {4.3: 'A+', 4.0: 'A', 3.7: 'A-', 3.3: 'B+', 3.0: 'B', 2.7: 'B-',
                  2.3: 'C+', 2.0: 'C', 1.7: 'C-', 1.3: 'D+', 1.0: 'D', 0.7: 'D-', 0.0: 'F'}
    closest_grade = min(grade_dict.keys(), key=lambda x: abs(x - numeric_grades))
    return grade_dict[closest_grade]

def dynamic_feature_selection(df, grade_column):
    features = [col for col in df.columns if col != grade_column and col != 'Student Name']
    return features

def preprocess_data(df, grade_column):
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df.fillna(0) 
    df[grade_column] = df[grade_column].fillna('')
    return df

def train_dynamic_model(df, grade_column):
    features = dynamic_feature_selection(df, grade_column)
    df = df[['Student Name', grade_column] + features]
    df_filled = preprocess_data(df, grade_column)
    df_filled = df_filled[df_filled[features + [grade_column]].notna().all(axis=1)]
    
    if df_filled.empty:
        raise ValueError("Training data is empty after preprocessing.")
    
    df_filled[grade_column] = df_filled[grade_column].map(convert_grades_to_numeric)
    
    X = df_filled[features]
    y = df_filled[grade_column]
    
    if X.empty or y.empty:
        raise ValueError("Feature or target data is empty.")
    
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if X_train.shape[0] == 0 or y_train.shape[0] == 0:
        raise ValueError("No samples available for training.")
    
    pipeline.fit(X_train, y_train)
    
    return pipeline

def predict_dynamic_grades(prediction_file_path, model, df, grade_column):
    prediction_data = pd.read_csv(prediction_file_path)
    prediction_data = preprocess_data(prediction_data, grade_column)
    
    if grade_column in prediction_data.columns:
        prediction_data[grade_column] = convert_grades_to_numeric(prediction_data[grade_column])
    
    features = dynamic_feature_selection(df, grade_column)
    X = prediction_data[features]
    
    predicted_grades_numeric = model.predict(X)
    predicted_grades = [convert_numeric_to_grades(grade) for grade in predicted_grades_numeric]
    
    prediction_data['Predicted Grade'] = predicted_grades
    
    if grade_column in prediction_data.columns:
        prediction_data[grade_column] = prediction_data[grade_column].apply(convert_numeric_to_grades)
    
    output_file_path = f"predicted_{os.path.basename(prediction_file_path)}"
    prediction_data.to_csv(output_file_path, index=False)
    print(f"Predicted grades saved to {output_file_path}")

def main():
    prediction_file_path = "newtest.csv"
    grade_column = "Current Grade"
    
    df = load_data("newTraining")
    
    try:
        timeline = infer_timeline(df)
        timeline_directory = os.path.join("newTraining", timeline)
        print(f"Timeline directory: {timeline_directory}")
        
        training_df = load_data(timeline_directory)
        
        model = train_dynamic_model(training_df, grade_column)
        
        predict_dynamic_grades(prediction_file_path, model, df, grade_column)
    
    except ValueError as e:
        print(f"Error: {e}")

main()
