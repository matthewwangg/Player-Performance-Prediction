import os
import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor, plot_importance
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from data_processing import find_path
from sklearn.model_selection import train_test_split

# Function to train all 4 models for each position
def train_models(dataframes, positions):

    trained_models = []

    for i in range(len(dataframes)):
        integer_columns = dataframes[i].select_dtypes(include=['int']).drop(columns=['total_points'])
        xgb_model = train_xgboost_model(integer_columns, dataframes[i]['total_points'], positions[i])
        trained_models.append(xgb_model)

    return trained_models

# Function to train XGBoost model with the predefined hyperparameters
def train_xgboost_model(X, y, position):

    # Define hyperparameters
    hyperparams = {
        'learning_rate': 0.1,
        'max_depth': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'n_estimators': 100,
        'alpha': 0.1,
        'lambda': 0.1
    }

    # Create XGBoost regressor with predefined hyperparameters
    xgb_model = XGBRegressor(**hyperparams)

    # Train the model
    xgb_model.fit(X, y)

    print(f"XGBoost Model trained for {position} position.")

    return xgb_model

# Function to evaluate XGBoost model and print MSE
def evaluate_model(model, X_test, y_test, position):

    # Get the predictions
    y_pred = model.predict(X_test)

    # Calculate MSE
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error for {position}: {mse}")

def visualize(models, output_dir, positions):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    visualizations = []

    for idx, model in enumerate(models):
        # Generate the feature importance plot
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_importance(model, ax=ax)

        # Save the plot as an image file
        image_path = os.path.join(output_dir, f"visualization_{positions[idx]}.png")
        fig.savefig(image_path, format='png')

        # Close the figure to release memory
        plt.close(fig)

        # Append the image path to the list of visualizations
        visualizations.append(image_path)

    return visualizations

# Function to perform hyperparameter tuning with cross-validation
def tune_hyperparameters(X, y):

    # Define the parameter grid
    param_grid = {
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 4],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'n_estimators': [100, 200]
    }

    # Create XGBoost regressor
    xgb_model = xgb.XGBRegressor()

    # Perform grid search with cross-validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=kfold)
    grid_search.fit(X, y)

    # Get the best parameters
    best_params = grid_search.best_params_
    print("Best Hyperparameters:", best_params)

    return best_params

# Function to train XGBoost model with the best hyperparameters
def train_xgboost_model_best(X, y, position):

    # Perform hyperparameter tuning
    best_params = tune_hyperparameters(X, y)

    # Create XGBoost regressor with best hyperparameters
    xgb_model = XGBRegressor(**best_params)

    # Train the model
    xgb_model.fit(X, y)

    print(f"XGBoost Model trained for {position} position.")

    return xgb_model

# Function to train Convolutional Neural Network
def train_neural_networks():
    # I'm just setting it up for future use

    players_df = pd.read_csv(find_path())

    # Assuming your dataframe is called 'df'
    X = players_df.drop("total_points", axis=1)  # Features (all columns except "total_points")
    y = players_df["total_points"]  # Target variable

    # Split data into training and testing sets (adjust test_size as needed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the CNN model
    model = keras.Sequential([
        # Input layer (adjust based on your data shape)
        keras.Input(shape=(img_height, img_width, channels)),  # Replace with your image dimensions and channels

        # Convolutional layers with ReLU activation
        keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(img_height, img_width, channels)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation="relu"),
        keras.layers.MaxPooling2D((2, 2)),

        # Flatten layer for dense connections
        keras.layers.Flatten(),

        # Dense layers with ReLU activation
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(64, activation="relu"),

        # Output layer for player points prediction (adjust units for other tasks)
        keras.layers.Dense(1)
    ])

    # Compile the model
    model.compile(loss="mse", optimizer="adam", metrics=["mae"])  # Mean squared error, mean absolute error

    # Data Augmentation (optional)
    datagen = ImageDataGenerator(rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2,
                                 zoom_range=0.2)

    # Train the model
    model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=10, validation_data=(X_test, y_test))

    # Evaluate the model
    loss, mae = model.evaluate(X_test, y_test)
    print(f"Loss: {loss}, MAE: {mae}")
