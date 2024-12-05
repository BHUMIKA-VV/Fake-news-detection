# %% [markdown]
# ![Screenshot 2024-12-04 214542.png](<attachment:Screenshot 2024-12-04 214542.png>)
# Import necessary libraries

# %%

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import string


# %% [markdown]
#   Importing datasets

# %%
df_true = pd.read_csv('C:/Users/Bhumika V/OneDrive/Desktop/New folder (2)/True.csv/True.csv')  # Replace with your true dataset file path
df_fake = pd.read_csv('C:/Users/Bhumika V/OneDrive/Desktop/New folder (2)/Fake.csv/Fake.csv')  # Replace with your false dataset file path

# Show the first 5 rows of both datasets
print("True Data Sample:")
df_true.head(5)

print("False Data Sample:")
df_fake.head(5)

# %% [markdown]
# Inspecting df_true data

# %%
df_true.head(5)



# %% [markdown]
# Inserting a column "class" as target feature

# %%
df_fake["class"] = 0
df_true["class"] = 1
df_fake.shape, df_true.shape

# %% [markdown]
# Removing last 10 rows for manual testing

# %%
df_fake_manual_testing = df_fake.tail(10)
for i in range(23480, 23470, -1):
    df_fake.drop([i], axis=0, inplace=True)

df_true_manual_testing = df_true.tail(10)
for i in range(21416, 21406, -1):
    df_true.drop([i], axis=0, inplace=True)
df_fake.shape, df_true.shape

# %% [markdown]
#  Adding "class" for manual testing

# %%
df_fake_manual_testing["class"] = 0
df_true_manual_testing["class"] = 1

# %% [markdown]
# Inspecting manual testing data

# %%
df_fake_manual_testing.head(10)

# %% [markdown]
# Inspecting df_true manual testing data

# %%
df_true_manual_testing.head(10)

# %% [markdown]
# Merging True and Fake Dataframes

# %%
df_merge = pd.concat([df_fake, df_true], axis=0)
df_merge.head(10)

# %% [markdown]
#  Removing unnecessary columns

# %%
df = df_merge.drop(["title", "subject", "date"], axis=1)
df.isnull().sum()

# %% [markdown]
#  Random Shuffling the dataframe

# %%
df = df.sample(frac=1)
df.head()

# %% [markdown]
# Resetting index

# %%
# Reset index and drop the 'index' column
df.reset_index(inplace=True)
df.drop(["index"], axis=1, inplace=True)

# View the columns of the dataframe
df.columns


# %% [markdown]
# Inspecting the dataframe after rese

# %%

df.head()

# %% [markdown]
#  Creating a function to process the texts

# %%
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

df["text"] = df["text"].apply(wordopt)

# %% [markdown]
#  Defining dependent and independent variables

# %%
x = df["text"]
y = df["class"]

# %% [markdown]
# Splitting Training and Testing Data

# %%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

# %% [markdown]
# Converting text to vectors using TfidfVectorizer

# %%
from sklearn.feature_extraction.text import TfidfVectorizer

vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

# %% [markdown]
# Logistic Regression Model

# %%
from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()
LR.fit(xv_train, y_train)
pred_lr = LR.predict(xv_test)
LR.score(xv_test, y_test)

# %% [markdown]
# Displaying classification report for Logistic Regression

# %%
print(classification_report(y_test, pred_lr))

# %% [markdown]
#  Decision Tree Classification

# %%
from sklearn.tree import DecisionTreeClassifier

DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)
pred_dt = DT.predict(xv_test)
DT.score(xv_test, y_test)

# %% [markdown]
# Displaying classification report for Decision Tree

# %%
print(classification_report(y_test, pred_dt))

# %% [markdown]
# Gradient Boosting Classifier

# %%
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score

# Initialize and train the GradientBoostingClassifier
GBC = GradientBoostingClassifier(n_estimators=100, random_state=0)
GBC.fit(xv_train, y_train)

# Make predictions and calculate accuracy
pred_gbc = GBC.predict(xv_test)
accuracy = accuracy_score(y_test, pred_gbc)

# Print accuracy and classification report
print(f"GradientBoostingClassifier Accuracy: {accuracy}")
print(classification_report(y_test, pred_gbc))


# %% [markdown]
#  Displaying classification report for Gradient Boosting

# %%
print(classification_report(y_test, pred_gbc))

# %% [markdown]
# Random Forest Classifier

# %%
from sklearn.ensemble import RandomForestClassifier

RFC = RandomForestClassifier(random_state=0)
RFC.fit(xv_train, y_train)
pred_rfc = RFC.predict(xv_test)
RFC.score(xv_test, y_test)



# %% [markdown]
#  Displaying classification report for Random Forest

# %%
print(classification_report(y_test, pred_rfc))

# %% [markdown]
# xgboost model

# %%
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Assuming `xv_train`, `y_train`, `xv_test`, and `y_test` are already defined

# Initialize the XGBoost classifier
xgboost_model = xgb.XGBClassifier(
    objective='binary:logistic', 
    eval_metric='logloss', 
    random_state=0,
    use_label_encoder=False
)

# Train the model
xgboost_model.fit(xv_train, y_train)

# Predict on the test set
pred_xgb = xgboost_model.predict(xv_test)

# Evaluate the model
accuracy = accuracy_score(y_test, pred_xgb)
cm_xgb = confusion_matrix(y_test, pred_xgb)
classification_rep = classification_report(y_test, pred_xgb)

# Print the results
print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(cm_xgb)
print("Classification Report:")
print(classification_rep)


# %% [markdown]
# Categorical Distribution (Bar Plot)

# %%
plt.figure(figsize=(8, 6))
sns.countplot(x='class', data=df)
plt.title('Class Distribution (True vs Fake)')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

# %% [markdown]
#  Support Vector Machine Model

# %%
from sklearn.svm import SVC  # Import the SVC class

# Create and train the model
svm = SVC()
svm.fit(xv_train, y_train)

# Make predictions
pred_svm = svm.predict(xv_test)

# Calculate accuracy
accuracy_svm = accuracy_score(y_test, pred_svm)

print("SVM Accuracy:", accuracy_svm)
 

# %% [markdown]
#  Naive Bayes Model

# %%
from sklearn.naive_bayes import MultinomialNB  # Import the MultinomialNB class

# Create and train the model
nb = MultinomialNB()
nb.fit(xv_train, y_train)

# Make predictions
pred_nb = nb.predict(xv_test)

# Calculate accuracy
accuracy_nb = accuracy_score(y_test, pred_nb)

print("Naive Bayes Accuracy:", accuracy_nb)


# %% [markdown]
# K-Nearest Neighbors Model

# %%
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(xv_train, y_train)
pred_knn = knn.predict(xv_test)
accuracy_knn = accuracy_score(y_test, pred_knn)

# %% [markdown]
#  Plotting the graphs (all in one cell)

# %%
import pandas as pd
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, accuracy_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.utils import resample

# Assuming `df` is your dataframe and `xv_train` is a sparse matrix
# Ensure `df` contains `class` and `text` columns.

# Setup Subplots
fig = make_subplots(
    rows=5, cols=3, 
    subplot_titles=(
        "Class Distribution", "Text Length by Class", "Model Accuracies",
        "Confusion Matrix (Logistic Regression)", "Precision-Recall Curve (SVM)", "ROC Curve (SVM)",
        "Top 20 Features (Decision Tree)", "Confusion Matrix (Naive Bayes)", "Confusion Matrix (Random Forest)",
        "K-Means Clustering", "PCA - Reduced Dimensionality", "Additional Visualization"
    )
)

# 1. Class Distribution
class_counts = df['class'].value_counts()
fig.add_trace(
    go.Bar(x=class_counts.index, y=class_counts.values, name="Class Distribution"),
    row=1, col=1
)

# 2. Text Length by Class
fig.add_trace(
    go.Box(x=df['class'], y=df['text'].apply(len), name="Text Length by Class"),
    row=1, col=2
)

# 3. Model Accuracies
model_accuracies = [accuracy_score(y_test, pred_lr), accuracy_svm, accuracy_score(y_test, pred_dt), 
                    accuracy_score(y_test, pred_gbc), accuracy_score(y_test, pred_rfc), accuracy_nb, accuracy_knn]
model_names = ['Logistic Regression', 'SVM', 'Decision Tree', 'Gradient Boosting', 'Random Forest', 'Naive Bayes', 'KNN']
fig.add_trace(
    go.Bar(x=model_names, y=model_accuracies, name="Model Accuracies"),
    row=1, col=3
)

# 4. Confusion Matrix (Logistic Regression)
cm_lr = confusion_matrix(y_test, pred_lr)
fig.add_trace(
    go.Heatmap(z=cm_lr, x=['Fake', 'True'], y=['Fake', 'True'], colorscale='Blues', name="Confusion Matrix (LR)"),
    row=2, col=1
)

# 5. Precision-Recall Curve (SVM)
precision, recall, _ = precision_recall_curve(y_test, pred_svm)
fig.add_trace(
    go.Scatter(x=recall, y=precision, mode='lines', name="Precision-Recall Curve (SVM)"),
    row=2, col=2
)

# 6. ROC Curve (SVM)
fpr, tpr, _ = roc_curve(y_test, pred_svm)
fig.add_trace(
    go.Scatter(x=fpr, y=tpr, mode='lines', name="ROC Curve (SVM)"),
    row=2, col=3
)

# 7. Top 20 Features (Decision Tree)
feature_importances = DT.feature_importances_
top_features = sorted(zip(feature_importances, range(len(feature_importances))), reverse=True)[:20]
fig.add_trace(
    go.Bar(
        x=[f"Feature {idx}" for _, idx in top_features],
        y=[importance for importance, _ in top_features],
        name="Top 20 Features (DT)"
    ),
    row=3, col=1
)

# 8. Naive Bayes Confusion Matrix
cm_nb = confusion_matrix(y_test, pred_nb)
fig.add_trace(
    go.Heatmap(z=cm_nb, x=['Fake', 'True'], y=['Fake', 'True'], colorscale='Reds', name="Confusion Matrix (NB)"),
    row=3, col=2
)

# 9. Random Forest Confusion Matrix
cm_rfc = confusion_matrix(y_test, pred_rfc)
fig.add_trace(
    go.Heatmap(z=cm_rfc, x=['Fake', 'True'], y=['Fake', 'True'], colorscale='Greens', name="Confusion Matrix (RFC)"),
    row=3, col=3
)

# 10. K-Means Clustering (Sampled Data)
sample_size = 5000
xv_train_sampled = resample(xv_train, n_samples=sample_size, random_state=42)
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(xv_train_sampled)
fig.add_trace(
    go.Scatter(x=kmeans.labels_, y=df['text'].iloc[:sample_size].apply(len), mode='markers', name="K-Means Clustering"),
    row=4, col=1
)

# 11. PCA - Reduced Dimensionality
pca = PCA(n_components=2, random_state=42)
xv_train_reduced = pca.fit_transform(xv_train_sampled)
fig.add_trace(
    go.Scatter(x=xv_train_reduced[:, 0], y=xv_train_reduced[:, 1], mode='markers', name="PCA Reduced Dimensionality"),
    row=4, col=2
)

# 12. Placeholder for Additional Visualization
fig.add_trace(
    go.Scatter(x=[0, 1], y=[1, 0], mode='lines', name="Placeholder"),
    row=4, col=3
)

# Update Layout
fig.update_layout(
    height=1500,
    width=1800,
    title_text="Fake News Detection Model Visualizations",
    showlegend=True
)

fig.show()



