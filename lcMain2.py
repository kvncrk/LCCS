import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Training the model

# Loading dataset
data = pd.read_csv('Plant_dataset.csv')

# Removing the whitespaces from column names
data.columns = data.columns.str.strip()

# Defining independent variables and dependent variable (Mood Score)
X = data[['averageMoisture', 'wavelength_nanometers', 'air_quality']].copy()
X.columns = X.columns.str.strip()
Y = data['well_score']

# Splitting the dataset into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Creating the Linear Regression model
model = LinearRegression()

# Fitting the model with the training data
model.fit(X_train, Y_train)

# Predicting mood scores for the test set
Y_pred = model.predict(X_test)

print("Multiple Linear Regression Model Complete!")

# Checking how well it worked
mse = mean_squared_error(Y_test, Y_pred)
print(f"Mean Squared Error: {mse}")

# Interpretation of Mean Squared Error, futher analyis
def interpret_mse(mse):
    if mse < 10:
        return "Excellent model accuracy."
    elif mse < 20:
        return "Good model accuracy."
    elif mse < 30:
        return "Average model accuracy."
    elif mse < 40:
        return "Below average model accuracy."
    else:
        return "Poor model accuracy!"

mse_remark = interpret_mse(mse)
print("How good is this model? ", mse_remark)

# Making a prediction using the model
def predict_well(averageMoisture, wavelength_nanometers, air_quality):
    df = pd.DataFrame([[averageMoisture, wavelength_nanometers, air_quality]],
                      columns=['averageMoisture', 'wavelength_nanometers', 'air_quality'])
    return model.predict(df)[0]
#--------------------------------------------------------------------------------------------------------------------------------

# Let the user enter their own 3 parameters
print("")
print("USER CHOOSES 3 INPUTS")
moisture = float(input("Enter selected moisture. Can be any anything from 1-800: "))
wavelength = int(input("Enter wavelength of LED. Can be integer of either 500/600/700:  "))
airquality = int(input("Enter air quality percentage. Can be any integer from 1-100: "))

predicted_well = predict_well(moisture, wavelength, airquality)
predicted_well = round(predicted_well, 2)
print("\n The predicted mental well-being score for the values entered is", predicted_well)
#-----------------------------------------------------------------------------------------------------------
# WHAT-IF Question 1
# What is will your mood be with low values of all three parameters?
print("-----------------------------------------------------------")
print("WHAT-IF QUESTION 1")
print("Let's test what the mood will be if overall poor and rare user care.")

# Low values for all 3 parameters
moisture = 200
wavelength = 500
airquality = 20

mood_if_littleCare = predict_well(moisture, wavelength, airquality)
mood_if_littleCare = round(mood_if_littleCare, 2)
print("\n The predicted passive care mental well-being score is", mood_if_littleCare)


# WHAT-IF Question 2
# What is will your mood be with high values of all three parameters?
print("-----------------------------------------------------------")
print("WHAT-IF QUESTION 2")
print("Let's test what the mental well-being will be if user ensured care of plant is always perfect")

# High values for all 3 parameters
moisture = 700
wavelength = 700
airquality = 78


mood_if_loadsCare = predict_well(moisture, wavelength, airquality)
mood_if_loadsCare = round(mood_if_loadsCare, 2)

# Cap the predicted score at 10 if it exceeds 10 (e.g. was 10.6)
mood_if_loadsCare = min(mood_if_loadsCare, 10)

print("\n The high care mood score is", mood_if_loadsCare)



# WHAT IF QUESTION 3
print("-----------------------------------------------------------")
print("ADDITIONAL WHAT-IF QUESTION 3")
print("Let's say a plant buddy user recently moved into a new house.\n They have just began painting the walls and have just varnished some of the wooden floor")
print("They have decided to keep their new plant in this environment, which is an environment unbeknownst to them, high in VOCs(indoor air pollution)")
print("They have nurtured the plant in all other aspects, but have kept it in this poor air quality unknowingly.")

moisture = 500
wavelength = 500
airquality = 33

mood_if_NewRoom = predict_well(moisture, wavelength, airquality)
mood_if_NewRoom = round(mood_if_loadsCare, 2)
print("\n The user who kept their plant buddy in a high VOC content rooms (bad air quality content) mood score is", mood_if_NewRoom)
print('This is high, which indicates the model predicted it did not have much impact on the users mental wellbeing')
print('↑ Scroll up for other 2 primary WHAT IF questions ↑')


#------------------------------------------------------------------------------------------
# AR3 Show Results of WHAT IF on a graph for Questions 1 & 2
# Advanced Requirement 3 showing graph comparision of first 2 what if questions
import matplotlib.pyplot as plt
import seaborn as sns # Import seaborn for additional styling
import numpy as np

# Names of variables and value
variable_names = ['Well-being if Poor Overall Care', 'Well-being if Overall Great Care']
values = [mood_if_littleCare, mood_if_loadsCare]

# Set seaborn style for better aesthetics
sns.set(style="whitegrid", context="notebook")

# Creating a bar chart with seaborn for better aesthetics
plt.figure(figsize=(10, 6))
bar_plot = sns.barplot(x=variable_names, y=values)

# Adding labels and title
plt.xlabel('Amount of Care', fontsize=14)
plt.ylabel('Well-being from 1-10', fontsize=14)
plt.title('Bar Chart of WHAT-IF Q1, Q2 Outcomes', fontsize=16)

# Adding data labels on top of the bars
for i, v in enumerate(values):
    bar_plot.text(i, v + 0.1, str(v), ha='center', va='bottom', fontsize=12)

# Show the plot
plt.show()
#-----------------------------------------------------------------------------

# Plotting scatter plot with regression line for well-being and average moisture
plt.figure(figsize=(12, 8))
plt.scatter(data['averageMoisture'], data['well_score'], color='blue', label='Data Points')
x_range = np.linspace(data['averageMoisture'].min(), data['averageMoisture'].max(), 100)
X_range = pd.DataFrame({'averageMoisture': x_range})
X_range['wavelength_nanometers'] = data['wavelength_nanometers'].mean()  # Adding mean value for wavelength
X_range['air_quality'] = data['air_quality'].mean()  # Adding mean value for air quality
y_range = model.predict(X_range)
plt.plot(x_range, y_range, color='green', linestyle='--', label='Regression Line')
plt.xlabel('Average Moisture', fontsize=14)
plt.ylabel('Well-being Score', fontsize=14)
plt.title('Scatter Plot with Regression Line for Average Moisture and Well-being Score', fontsize=16)
plt.legend()
# Show the plot
plt.show()


# -------------------------------------------------
# #  Plotting scatter plot with regression line for well-being and air quality
# 
# plt.figure(figsize=(12, 8))
# plt.scatter(data['air_quality'], data['well_score'], color='blue', label='Data Points')
# x_range = np.linspace(data['air_quality'].min(), data['air_quality'].max(), 100)
# x_range = pd.DataFrame({'air_quality': x_range})
# x_range['averageMoisture'] = data['averageMoisture'].mean()  # Adding mean value for average moisture
# x_range['wavelength_nanometers'] = data['wavelength_nanometers'].mean()  # Adding mean value for wavelength
# x_range = x_range[['averageMoisture', 'air_quality', 'wavelength_nanometers']]
# y_range = model.predict(X_range)
# 
# plt.plot(x_range, y_range, color='green', linestyle='--', label='Regression Line')
# plt.xlabel('Air Quality', fontsize=14)
# plt.ylabel('Well-being Score', fontsize=14)
# plt.title('Scatter Plot with Regression Line for Air Quality and Well-being Score (not a strong linear relationship)', fontsize=16)
# plt.legend()
# # Show the plot
# plt.show()
# -----------------------------------

# Plotting scatter plot with regression line for all variablesimport matplotlib.pyplot as plt


# Create a figure and a 3D axis
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
ax.scatter(data['averageMoisture'], data['wavelength_nanometers'], data['well_score'], c='blue', marker='o')

# Set labels and title
ax.set_xlabel('Average Moisture')
ax.set_ylabel('Wavelength (nanometers)')
ax.set_zlabel('Well-being Score')
ax.set_title('3D Scatter Plot of LED Wavelength, Average Moisture, and Well-being Score')

# Show the plot
plt.show()