from scipy.spatial.distance import pdist
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np
from scipy.spatial.distance import cdist
from numpy.linalg import lstsq
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

def load_dataset():
    df = pd.read_csv(
        "AirQualityUCI.csv",
        sep = ";",
        decimal = ","
    )
    return df


# print(df.shape)
# print(df.head())
# print(df.columns)
# df.info()
# print(df.isnull().sum())

def clean_dataset(df):
    # this will remove the completly empty columns
    df = df.dropna(axis=1, how="all")

    #remove completely empty rows
    df = df.dropna(how="all") #by defaut axis = 0 for rows all means if all are NaN then delete the row
                              #we use any and it will detete the row if any entry in row is NaN
    df = df.replace(-200, float("nan")) # I am doing this because uci air quality dataset use -200 as missing

    return df




# print("Shape after cleaning", df.shape)
# print("Missing Values : ")
# print(df.isnull().sum())
# print("Missing CO measurements:", df["CO(GT)"].isnull().sum())
# print("Total Observations:", len(df))


def prepare_features(df):
    #remove rows where target co concentration is missing
    df = df.dropna(subset=["CO(GT)"])

    #select input features
    features = [
        "PT08.S1(CO)",
        "PT08.S2(NMHC)",
        "PT08.S3(NOx)",
        "PT08.S4(NO2)",
        "PT08.S5(O3)",
        "T",
        "RH",
        "AH"
    ]

    X = df[features] #inputs
    y = df["CO(GT)"] #expected output

    return X, y


def split_dataset(X, y):
    
    #use the first 80% for training
    split_index = int(len(X)*0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    return X_train, X_test, y_train, y_test


from sklearn.impute import SimpleImputer

def impute_missing_values(X_train, X_val, X_test):
    imputer = SimpleImputer(strategy="median")

    #calculate medians using training data
    X_train = imputer.fit_transform(X_train)

    X_val = imputer.transform(X_val)
    #same for testing data
    X_test = imputer.transform(X_test)

    return X_train, X_val, X_test

def scale_features(X_train, X_val, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    return X_train, X_val, X_test

def preprocess_final_data(X_train, X_test):
    imputer = SimpleImputer(strategy="median")

    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test


def find_centers(X_train, k):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_train)
    centers = kmeans.cluster_centers_
    return centers

def calculate_sigma(centers):
    distances = pdist(centers)
    sigma = np.median(distances)
    return sigma

def calculate_activations(X, centers, sigma):
    distances = cdist(X, centers)
    phi = np.exp(-(distances**2)/(2*sigma**2))
    return phi


def train_rbf(phi, y_train):
    weights = lstsq(phi, y_train, rcond=None)[0]
    return weights

def predict(X, centers, sigma, weights):
    phi = calculate_activations(X, centers, sigma)
    predictions = phi@weights
    return predictions

def evaluate_model(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    return mae, rmse, r2

def tune_sigma(X_train, y_train, X_val, y_val, centers, initial_sigma):
    sigma_values = np.linspace(
        0.5*initial_sigma,
        1.5*initial_sigma,
        11
    )
    best_sigma = None
    best_rmse = float('inf')

    for sigma in sigma_values:
        phi_train = calculate_activations(X_train, centers, sigma)
        weights = train_rbf(phi_train, y_train)
        y_pred = predict(X_val, centers, sigma, weights)
        mae, rmse, r2 = evaluate_model(y_val, y_pred)

        if(rmse < best_rmse):
            best_rmse = rmse
            best_sigma = sigma
    
    fine_values = np.linspace(
        0.8*best_sigma,
        1.2*best_sigma,
        11
    )

    for sigma in fine_values:
        phi_train = calculate_activations(X_train, centers, sigma)
        weights = train_rbf(phi_train, y_train)
        y_pred = predict(X_val, centers, sigma, weights)
        mae, rmse, r2 = evaluate_model(y_val, y_pred)

        if(rmse < best_rmse):
            best_rmse = rmse
            best_sigma = sigma
    
    return best_sigma, best_rmse

def tune_centers(X_train, y_train, X_val, y_val):
    k_values = [10, 25, 50, 100]
    best_k = None
    best_sigma = None
    best_initial_sigma = None
    best_rmse = float('inf')

    for k in k_values:
        centers = find_centers(X_train, k)
        initial_sigma = calculate_sigma(centers)
        sigma, rmse = tune_sigma(X_train, y_train, X_val, y_val, centers, initial_sigma)

        if(rmse < best_rmse):
            best_rmse = rmse
            best_k = k
            best_sigma = sigma
            best_initial_sigma = initial_sigma
    return best_k, best_sigma, best_initial_sigma, best_rmse


def split_validation(X_train, y_train):
    split_index = int(len(X_train)*.80)
    X_subtrain = X_train[:split_index]
    X_val = X_train[split_index:]
    y_subtrain = y_train.iloc[:split_index]
    y_val = y_train.iloc[split_index:]
    return X_subtrain, X_val, y_subtrain, y_val

df = load_dataset()
df = clean_dataset(df)
X, y = prepare_features(df)
X_train, X_test, y_train, y_test = split_dataset(X, y)
X_test_raw = X_test.copy()

X_subtrain, X_val, y_subtrain, y_val = split_validation(X_train, y_train)

X_subtrain, X_val, _ = impute_missing_values(X_subtrain, X_val, X_val)

X_subtrain, X_val, _ = scale_features(X_subtrain, X_val, X_val)



best_k, best_sigma, best_initial_sigma, best_rmse = tune_centers(X_subtrain, y_subtrain, X_val, y_val)



X_train_final, X_test_final = preprocess_final_data(X_train, X_test_raw)

centers_final = find_centers(X_train_final, best_k)

initial_sigma_final = calculate_sigma(centers_final)

sigma_final = initial_sigma_final*(best_sigma/best_initial_sigma)


phi_final = calculate_activations(X_train_final, centers_final, sigma_final)

weights_final = train_rbf(phi_final, y_train)


joblib.dump({
    "centers": centers_final,
    "sigma": sigma_final,
    "weights": weights_final,
    "best_k": best_k
}, "rbf_model.pkl")

print("RBF model saved successfully")


y_pred = predict(X_test_final, centers_final, sigma_final, weights_final)

mae, rmse, r2 = evaluate_model(y_test, y_pred)

print("\nFinal Test Results")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)

# print("DatasetShape:", df.shape)
# print("\nInput Shape:", X.shape)
# print("\nTarget Shape:", y.shape)
# print("\nFirst 5 input observations:")
# print(X.head())
# print("\nFirst 5 target observations:")
# print(y.head())

# print("Training input:", X_train.shape)
# print("Testing input:", X_test.shape)
# print("Training target:", y_train.shape)
# print("Testing target:", y_test.shape)
# print("Missing Values in Training:", pd.isna(X_train).sum())
# print("Missing Values in Testing:", pd.isna(X_test).sum())

# print("Centers shape:", centers.shape)
# print("Initial Sigma:", sigma)

# print("Phi shape", phi.shape)
# print("Weights shape:", weights.shape)
# print("Predictions Shape", y_pred.shape)
# print("MAE:", mae)
# print("RMSE:", rmse)
# print("R2 score:", r2)

# print("Subtraining input:", X_subtrain.shape)
# print("Validation input:", X_val.shape)
# print("Test input:", X_test.shape)
# print("Subtraining target:", y_subtrain.shape)
# print("Validation target:", y_val.shape)
# print("Test target:", y_test.shape)

# print("Best K:", best_k)
# print("Best Sigma:", best_sigma)
# print("Best validation RMSE:", best_rmse)


# print("Final training input shape:", X_train_final.shape)
# print("Final activation matrix shape:", phi_final.shape)
# print("Final weights shape:", weights_final.shape)
# print("Test predictions shape:", y_pred.shape)
# print("Test target shape:", y_test.shape)


import matplotlib.pyplot as plt
plt.figure(figsize = (8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)

#perfect prediction reference line
min_value = min(y_test.min(), y_pred.min())
max_value = max(y_test.max(), y_pred.max())
plt.plot([min_value, max_value], [min_value, max_value], "r--")

plt.xlabel("Actual CO")
plt.ylabel("Predicted CO")
plt.title("RBF Neural Network: Actual vs Predicted CO")
plt.tight_layout()
plt.show()

residuals = y_test - y_pred

plt.figure(figsize = (8, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='red', linestyle="--")

plt.xlabel("Predicted CO")
plt.ylabel("Residual (Actual - Predicted)")
plt.title("RBF Neural Network: Residual Plot")
plt.tight_layout()
plt.show()



saved_model = joblib.load("rbf_model.pkl")

print("Saved model keys:", saved_model.keys())
print("Best K:", saved_model["best_k"])
print("Sigma:", saved_model["sigma"])
print("Centers shape:", saved_model["centers"].shape)
print("Weights shape:", saved_model["weights"].shape)
