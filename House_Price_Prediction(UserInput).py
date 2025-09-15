import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
import tkinter as tk
from tkinter import simpledialog, messagebox

# Load data
data = pd.read_csv("./data/house_price_train.csv")

# Check Data info
data.info()
print(data.head())
print(data.columns)


# Split data
Features = ["date","condition","sqft_living","sqft_lot","sqft_lot15" ,"floors", "bedrooms", "bathrooms","waterfront","sqft_above", "sqft_basement","yr_built", "yr_renovated","zipcode","grade","lat", "long","view"]
x = data[Features]
y = data["price"]

    
x_train, x_valid, y_train, y_valid = train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=0)

# Initialize the Neural Network model
nn_model = MLPRegressor(hidden_layer_sizes=(250 , 150), max_iter=3000, random_state=1, learning_rate_init=0.01, batch_size=64, alpha=0.001)

# Fit the Neural Network model
nn_model.fit(x_train, y_train)

# Predict on validation set using the Neural Network model
y_nn_predictions = nn_model.predict(x_valid)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_valid_scaled = scaler.transform(x_valid)
# Calculate accuracy metrics for Neural Network
mae_nn = mean_absolute_error(y_valid, y_nn_predictions)
mse_nn = mean_squared_error(y_valid, y_nn_predictions)
r2_nn = r2_score(y_valid, y_nn_predictions)
def plot_training_curve(model):
    # Extract the training loss curve
    loss_curve = model.loss_curve_
    
    # Plot the training loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(loss_curve, marker='o', linestyle='-', color='b')
    plt.title('Training Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()
# Plot the training curve
plot_training_curve(nn_model)

    
print(f"Neural Network MAE: {mae_nn}")
print(f"Neural Network MSE: {mse_nn}")
print(f"Neural Network R²: {r2_nn}")


#
# Function to predict and compare house prices based on multiple user inputs 
def predict_multiple_user_inputs(model, features):
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    house_data_list = []
    num_houses = simpledialog.askinteger("Input Number of Houses", "How many houses do you want to compare?")
    
    if num_houses is None or num_houses < 1:
        messagebox.showwarning("Invalid Input", "Please enter a valid number of houses.")
        return

    for i in range(num_houses):
        user_data = {}
        for feature in features:
            value = simpledialog.askfloat(f"House {i+1}: Input {feature}", f"Please enter the value for {feature}:")
            if value is None:  # User pressed "Cancel"
                messagebox.showinfo("Cancelled", "Input process was cancelled.")
                return  # Exit the function if "Cancel" is pressed
            user_data[feature] = value
        house_data_list.append(user_data)

    # Convert user input into a DataFrame
    user_input_df = pd.DataFrame(house_data_list)

    # Predict using the provided model
    predictions = model.predict(user_input_df)
    
    # Display entered user data and predictions
    results = []
    for i, prediction in enumerate(predictions):
        house_info = "\n".join([f"{feature}: {house_data_list[i][feature]}" for feature in features])
        results.append(f"House {i+1}:\n{house_info}\nPredicted price: {prediction:,.2f} triệu VND\n")

    messagebox.showinfo("Comparison Result", "\n".join(results))

    # Create DataFrame to display both input and predicted price
    comparison_df = pd.DataFrame(house_data_list)
    comparison_df['Predicted Price (triệu VND)'] = predictions

    # Print the table to the console (optional)
    print(comparison_df)

    # Plot the comparison of predicted prices using correct column names
    comparison_df.plot(kind='bar', y='Predicted Price (triệu VND)', legend=False, figsize=(10, 6))
    plt.title('Comparison of Predicted House Prices')
    plt.xlabel('House')
    plt.ylabel('Predicted Price ($)')
    plt.xticks(rotation=0)
    plt.show()

# Call the function to compare house prices based on user input
print("So sánh giá nhà dựa trên dữ liệu nhập từ nhiều căn nhà:")
predict_multiple_user_inputs(nn_model, Features)