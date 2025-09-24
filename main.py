"""This Python source file was created as a testing ground and template for the Jupyter Notebook."""

# import libraries
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# -------------- BUILD THE MODEL -------------------

# load dataset
df = pd.read_csv("student_habits_performance.csv")

# drop the unnecessary 'student_id' column
df = df.drop(columns=['student_id'])

# define the target column as 'exam_score'
target_column = 'exam_score'
x = df.drop(columns=[target_column])
y = df[target_column]

# identify the categorical (non-continuous) columns
categorical_columns = ['gender',
                       'part_time_job',
                       'diet_quality',
                       'parental_education_level',
                       'internet_quality',
                       'extracurricular_participation'
                       ]

# fill empty columns with placeholders
x[categorical_columns] = x[categorical_columns].fillna('N/A')
x = x.fillna(x.mean(numeric_only=True))

# -------------- TRAIN THE MODEL -------------------

# train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# initialize and train the CatBoostRegressor
model = CatBoostRegressor(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    cat_features=categorical_columns,
    verbose=False,
)

# fit the training data to the model
model.fit(x_train, y_train)

# make a prediction
y_pred = model.predict(x_test)

# evaluate the model's accuracy (using mean squared error and R² score)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# print evaluation results to the console
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# -------------- DISPLAY VISUALIZATIONS OF THE DATA -------------------

# save the name and relative importance of each feature in variables for use in plot generation
feature_importance = model.get_feature_importance()
feature_names = x.columns

# combine "feature_importance" and "feature_names" into their own DataFrame for easy sorting/selection
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})

"""
# generate a histogram graph showcasing the relative importance of each feature
plt.figure(figsize=(10, 6))
plt.barh(feature_names, feature_importance)
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.gca().invert_yaxis()
plt.show()
"""

# generate a pie chart showcasing the relative importance of the top features
top_n = 6
importance_df = importance_df.sort_values(by='Importance', ascending=False).head(top_n)

plt.figure(figsize=(8, 8))
plt.pie(
    importance_df['Importance'],
    labels=importance_df['Feature'],
    autopct='%1.1f%%',
    startangle=140,
    shadow=True,
)

plt.title(f'Top {top_n} Most Important Features')
plt.axis('equal')
plt.tight_layout()
plt.show()

# generate a scatter plot with a regression line for study hours (the most important feature)
plt.figure(figsize=(8, 6))
sns.regplot(x='study_hours_per_day', y='exam_score', data=df, scatter_kws={'alpha': 0.6}, line_kws={'color': 'red'})
plt.title('Study Hours vs. Exam Score (with Trend Line)')
plt.xlabel('Study Hours')
plt.ylabel('Exam Score')
plt.grid(True)
plt.tight_layout()
plt.show()

# generate a residual distribution histogram to visualize the model's prediction error
residuals = y_test - y_pred
plt.figure(figsize=(8, 5))
sns.histplot(residuals, kde=True)
plt.title('Residual Distribution')
plt.xlabel('Prediction Error')
plt.grid(True)
plt.show()


# -------------- MAKE PREDICTIONS FOR NEW INPUT DATA -------------------

# function to collect input from the user in order to make predictions
def predict_exam_score(model):
    while True:
        try:
            # collect data from user (with built-in error correction)
            print('Please enter the following student details to predict an exam score:')
            age = float(input('Age: '))
            gender = input('Gender (Male/Female): ').strip().lower()
            while gender not in ['male', 'female']:
                gender = input('ERROR - Please enter "Male" or "Female": ').strip().lower()
            gender = gender.capitalize()
            study_hours_per_day = float(input('Average number of study hours per day: '))
            social_media_hours = float(input('Average number of social media hours per day: '))
            netflix_hours = float(input('Average number of streaming hours (i.e., Netflix) per day: '))
            part_time_job = input('Has part-time job? (Yes/No): ').strip().lower()
            while part_time_job not in ['yes', 'no']:
                part_time_job = input('ERROR - Please enter "Yes" or "No": ').strip().lower()
            part_time_job = part_time_job.capitalize()
            attendance_percentage = float(input('Class attendance %: '))
            while attendance_percentage > 100 or attendance_percentage < 0:
                attendance_percentage = float(input('ERROR - Please enter a number between 0 and 100: '))
            sleep_hours = float(input('Average number of sleep hours per day: '))
            diet_quality = input('Diet quality (Poor/Fair/Good): ').strip().lower()
            while diet_quality not in ['poor', 'fair', 'good']:
                diet_quality = input('ERROR - Please enter "Poor", "Fair", or "Good": ').strip().lower()
            diet_quality = diet_quality.capitalize()
            exercise_frequency = int(input('Exercise frequency (times per week): '))
            parental_education_level = input('Parental education level (High School/Bachelor/Master): ').strip().lower()
            while parental_education_level not in ['high school', 'bachelor', 'master']:
                parental_education_level = input(
                    'ERROR - Please enter "High School", "Bachelor", or "Master": ').strip().lower()
            if parental_education_level == 'high school':
                parental_education_level = 'High School'
            else:
                parental_education_level = parental_education_level.capitalize()
            internet_quality = input('Internet quality (Poor/Average/Good): ').strip().lower()
            while internet_quality not in ['poor', 'average', 'good']:
                internet_quality = input('ERROR - Please enter "Poor", "Average", or "Good": ').strip().lower()
            internet_quality = internet_quality.capitalize()
            mental_health_rating = int(input('Mental Health Rating (1-10): '))
            while mental_health_rating > 10 or mental_health_rating < 1:
                mental_health_rating = int(input('ERROR - Please enter a number between 1 and 10: '))
            extracurricular_participation = input('Participates in extracurriculars? (Yes/No): ').strip().lower()
            while extracurricular_participation not in ['yes', 'no']:
                extracurricular_participation = input('ERROR - Please enter "Yes" or "No": ').strip().lower()
            extracurricular_participation = extracurricular_participation.capitalize()
        except ValueError:
            print('ERROR - Please enter numeric values when required.\n')
            print('Restarting...\n')
            continue

        # organize user input into a dictionary
        user_input = {
            'age': age,
            'gender': gender,
            'study_hours_per_day': study_hours_per_day,
            'social_media_hours': social_media_hours,
            'netflix_hours': netflix_hours,
            'part_time_job': part_time_job,
            'attendance_percentage': attendance_percentage,
            'sleep_hours': sleep_hours,
            'diet_quality': diet_quality,
            'exercise_frequency': exercise_frequency,
            'parental_education_level': parental_education_level,
            'internet_quality': internet_quality,
            'mental_health_rating': mental_health_rating,
            'extracurricular_participation': extracurricular_participation
        }

        # convert the input to the dataframe
        df_input = pd.DataFrame([user_input])

        # fill in the missing categorical values
        df_input[categorical_columns] = df_input[categorical_columns].fillna('N/A')

        # Predict exam score
        predicted_score = model.predict(df_input)[0]
        print(f'\nPredicted score: {predicted_score:.2f}')

        # exit loop or try again
        another = input("Would you like to predict another student's score? (Yes/No): ").strip().lower()
        if another != 'yes':
            print('Have a nice day :)')
            break


# call the predict_exam_score() method to run the user interface
predict_exam_score(model)
