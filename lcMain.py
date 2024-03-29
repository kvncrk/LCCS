#Prompt the user to input mood and anxiety levels
def validate_mood_and_anxiety(prompt):
    while True:
        try:
            value = float(input(prompt))
            if 1 <= value <= 10:
                return value
            else:
                raise ValueError
        except ValueError:
            print("Error: Input must be a number between 1 and 10")

mood_input = validate_mood_and_anxiety('From 1-10, rate your mood for the last day: ')
anxiety_input = validate_mood_and_anxiety('From 1-10, rate your calmness level, 1 being most anxious: ')

# Calculate the mean mood score
from statistics import mean
well_score = mean([mood_input, anxiety_input])
well_score = round(well_score, 1)

# Display the assigned mood score for the day
print('So your assigned mental well-being score for the day is', well_score)
#-----------------------------------------------------------

# Initialize an empty list to store soil moisture readings
moistureList =[]

import serial

# Create a serial connection to the microbit
ser = serial.Serial()
ser.baudrate = 115200
ser.port = "COM3" 

# Open the serial connection
ser.open()

# Iterate three times to collect soil moisture readings
for x in range(3):
                  
    # First take in all the data readings and assign it to this variable
    microbitdata = str(ser.readline())
    
    # Get second bit onwards, call that soil_moisture
    soil_moisture = microbitdata[2:]
    
    # Remove any spaces
    soil_moisture = soil_moisture.replace(" ","")
    
    # Remove any apostrophies
    soil_moisture = soil_moisture.replace("'","")
    
    # Remove it 
    soil_moisture = soil_moisture.replace("\\r\\n","")
    
    # Change to float value
    soil_moisture = float(soil_moisture)
    
    # Print it 
    print(soil_moisture)
    
    # Validation on data ; BR2 (Basic Requirement 2)
    if type(soil_moisture) == float:
        
        # If float then append to the list at the start
        moistureList.append(soil_moisture)                

# Print out list from start with the values which if were float, have been added
print('The moisture readings values which have come through are', moistureList)

# Getting single value, average value of list
averageMoisture = round(mean(moistureList),2)
print('The assigned moisture value for today is', averageMoisture)

# Analysis component based on moisture value, calculating information and giving user chance to base future decisions with their care on this 
if averageMoisture > 1000:
    print("Your plant is thriving! You're doing well in taking care of it.")
elif averageMoisture < 500:
    print("Your plant needs more care. Consider watering it to improve its well-being.")
else:
    print("Your plant is in a moderate condition. Keep an eye on its moisture levels.")

# User inputs LED zipstick display colour, depending on its colour, the code assigns the corresponding wavelength in nanometers for it

def validate_led_color():
    while True:
        try:
            value = int(input("Input corresponding number to LED colour used today for the system: \n 1 = red \n 2 = green \n 3 = yellow \n: "))
            if value in {1, 2, 3}:
                return value
            else:
                raise ValueError
        except ValueError:
            print("Error: Input must be 1, 2, or 3")

zip_colour = validate_led_color()

# Map the color input to wavelength
if zip_colour == 1:
    wavelength_nanometers = 700
elif zip_colour == 2:
    wavelength_nanometers = 500
elif zip_colour == 3:
    wavelength_nanometers = 600
else:
    print("Invalid input")

# Now can use the 'wavelength_nanometers' variable as needed in the rest of code.
print(f"The wavelength of the greenhouse kits LEDs is {wavelength_nanometers} nanometers.")

# User inputs IAQ % from microbits OLED display
def validate_air_quality():
    while True:
        try:
            value = int(input("From your second microbits oled display, input the air quality index (IAQ %), (If it's 100 percent, input 100): "))
            if 0 <= value <= 100:
                return value
            else:
                raise ValueError
        except ValueError:
            print("Error: Input must be an integer between 0 and 100")

air_quality = validate_air_quality()
print(f"The air quality index percentage is {air_quality}.")

# Prints out all the different data which have been acquired and which will be wrote to csv file
print(f"well-being score = {well_score}, moisture = {averageMoisture}, wavelength = {wavelength_nanometers}, air quality = {air_quality}")
#-----------------------------------------------------------------------------------------------------

# Using csv module
import csv

# Appending results to csv file
path = "'Plant_dataset.csv'"  #your file name, will create or overwrite.
f = open(path, "a", newline='')

csver = csv.writer(f)

# Writing rows to csv file
csver.writerow([averageMoisture, wavelength_nanometers, air_quality, well_score, mood_input])

print("I have added the following data to your csv")
print([averageMoisture, wavelength_nanometers, air_quality, well_score, mood_input])
print("")
f.close()

# Previewing what is now in the csv file using pandas

import pandas as pd

df = pd.read_csv('Plant_dataset.csv')

print(df)
