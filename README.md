# Disaster Response Pipeline Project


### Context
In this project, I will extract data set containing real messages that were sent during disaster events. I will utlize a machine learning pipeline to categorize these events so that we can send the messages to an appropriate disaster relief agency.

The project includes a web app where an emergency worker can input a new message and get classification results in several categories. 


### Instructions
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage

### Flask Web App
To run our web app python run.py Run env | grep WORK command is used to obtain space-id and workspace-id. You can reach the web app from 'https://view6914b2f4-3001.udacity-student-workspaces.com/'

Below are 2 screenshot of the web app.

![](/screenshot.png)

[](/screenshot2.png)


### Files in the repository

Here is an overview of the directory of this project:

```
- Root Directory
    - app
        - templates
        - run.py
        - templates.tar.gz
    - data
            - process_data.py
            - disaster_categories.csv
            - disaster_messages.csv
    - models
        - train_classifier.py
    - models
    - README.md
```

In the App folder, `run.py` is the Flask file that runs app.
the data folder contains the data to process and a py script that process the data.
The models folder contrain the script to train the model and a pickle file that contains the saved model. You can read more details below:


### Project Componenets

1. ETL Pipeline
process_data.py, write a data cleaning pipeline that:

- Loads the messages and categories datasets
- Merges the two datasets
- Cleans the data
- Stores it in a SQLite database

2. ML Pipeline
train_classifier.py, write a machine learning pipeline that:

- Loads data from the SQLite database
- Splits the dataset into training and test sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs results on the test set
- Exports the final model as a pickle file
