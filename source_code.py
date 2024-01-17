
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report

# Read the dataset from the CSV file
data = pd.read_csv('Train2021.csv')
dataTestOrg = pd.read_csv('Test Data.csv')
dataTest = dataTestOrg
# Select relevant features for prediction
features = ['hrv_mean_nni', 'hrv_median_nni', 'hrv_range_nni', 'hrv_sdsd', 'hrv_rmssd', 'hrv_nni_50', 'hrv_pnni_50',
            'hrv_nni_20', 'hrv_pnni_20', 'hrv_cvsd', 'hrv_sdnn', 'hrv_cvnni', 'hrv_mean_hr', 'hrv_min_hr',
            'hrv_max_hr', 'hrv_std_hr', 'hrv_total_power', 'hrv_vlf', 'hrv_lf', 'hrv_hf', 'hrv_lf_hf_ratio',
            'hrv_lfnu', 'hrv_hfnu', 'hrv_SD1', 'hrv_SD2', 'hrv_SD2SD1', 'hrv_CSI', 'hrv_CVI', 'hrv_CSI_Modified',
            'hrv_mean', 'hrv_std', 'hrv_min', 'hrv_max', 'hrv_ptp', 'hrv_sum', 'hrv_energy', 'hrv_skewness',
            'hrv_kurtosis', 'hrv_peaks', 'hrv_rms', 'hrv_lineintegral', 'hrv_n_above_mean', 'hrv_n_below_mean',
            'hrv_n_sign_changes', 'hrv_iqr', 'hrv_iqr_5_95', 'hrv_pct_5', 'hrv_pct_95', 'hrv_entropy',
            'hrv_perm_entropy', 'hrv_svd_entropy', 'Temperature', 'Humidity', 'PMV','PDD', 'Gender', 'Age',
            'X_axis', 'Y_axis', 'Z_axis']

#print(len(features))

target = 'Personal Thermal Assessment'

# Encode categorical variables
encoder = LabelEncoder()
data['Gender'] = encoder.fit_transform(data['Gender'])
dataTest['Gender'] = encoder.fit_transform(dataTest['Gender'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=42)

#print(X_train.shape)
#print(X_test.shape)



#testTarget="Personal Thermal Assessment"
dataTest = dataTest[features]

#dataTest = dataTest.drop(columns=target)

#print("real before FS",dataTest.shape)

# Create a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)


# Train the model
model.fit(X_train, y_train)


# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model
#print(classification_report(y_test, predictions))



# feature selection
feature_selector = SelectFromModel(model, prefit=True)

X_train_new = feature_selector.transform(X_train)
X_test_new = feature_selector.transform(X_test)
### test data #####
X_test_real = feature_selector.transform(dataTest)

#print(X_train_new.shape)
#print(X_test_new.shape)
#print("real: ",X_test_real.shape)

# Train the model
model.fit(X_train_new, y_train)

# Make predictions on the test set
#predictions_new = model.predict(X_test_new)
predictions_Test = model.predict(X_test_real)

# Evaluate the model
#print(classification_report(y_test, predictions_new))

#print("Prediction Test: ",predictions_Test )


submissionData = dataTestOrg
submissionData['Personal Thermal Assessment'] = predictions_Test
submissionData.to_csv('entireOutput.csv', index=False)
# Select specific columns
selected_columns = ['datetime', 'subject','Personal Thermal Assessment']
df_selected = submissionData[selected_columns]

# Save the selected columns to a CSV file
df_selected.to_csv('output.csv', index=False)








