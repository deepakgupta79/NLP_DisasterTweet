# Import necessary libraries
import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Number of samples
num_samples = 1000

# Generate random text data

text_data = np.random.choice(
    ['Help! The fire is spreading quickly in the forest near Main Street. Urgent assistance needed! 🔥🌲',
     'Just witnessed a massive earthquake. Everything is shaking! #earthquake',
     'False alarm, folks. The disaster was just a loud construction site next door. Sorry for the confusion!',
     'Beautiful day for a picnic by the river! 🌳☀️ #NotADisaster',
     'Oh no, my sandwich just fell apart! What a disaster! 🥪😱 #LunchFail',
     'Major flooding reported in the downtown area. Evacuation underway. Stay safe, everyone!',
     'Sarcasm alert: Just lost my pen. World-ending crisis, obviously. 🖊️😂',
     'Tornado warning issued for the southern region. Seek shelter immediately. #TornadoAlert',
     'Heavy snowfall in the mountainous region. Roads are icy and dangerous. #Snowstorm',
     'The concert got canceled? This is the worst disaster ever! 🎤😭 #Sarcasm'],
    size=num_samples)

# Generate random target labels (0 or 1)
target_labels = np.random.choice([0, 1], size=num_samples)

# Create DataFrame
df = pd.DataFrame({'text': text_data, 'target': target_labels})

# Save the DataFrame to a CSV file
df.to_csv('synthetic_dataset.csv', index=False)

# Display column names and information about the DataFrame
print("Column Names:", df.columns)
print("DataFrame Information:")
print(df.info())
