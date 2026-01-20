import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import homogeneity_completeness_v_measure
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC





X=np.load("/content/drive/MyDrive/Colab Notebooks/data/X_tp1.npy")
Y_raw=np.load("/content/drive/MyDrive/Colab Notebooks/data/Y_tp1.npy")

print(f"Shape of inputs: {X.shape}")
print(f"Shape of labels: {Y_raw.shape}")


plt.plot(X[0])
plt.show()


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(['A','L','N','R','V'])
Y = le.transform(Y_raw)
print(Y[0:10])

for i in range(0,5):
  S = X[(Y==i)]
  moyenne = np.mean(S,axis = 0)
  plt.plot(moyenne)
  plt.show()


data = X
scaler = preprocessing.StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
plt.plot(X_scaled[0])
plt.plot(X[0])
plt.show()

pca = PCA(n_components  = 180)
pca.fit(X_scaled)
PCA(n_components = 180)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Nombre de dimensions")
plt.ylabel("Ratio de variance expliquée")
plt.show()

nb_comp = np.where(np.cumsum(pca.explained_variance_ratio_)>0.95)[0][0]+1
print(nb_comp)

pca = PCA(n_components = nb_comp)
pca.fit(X_scaled)
X_projected = pca.transform(X_scaled)
print("Shape des données :", X_projected.shape)


pca = PCA(n_components = nb_comp)
pca.fit(X_scaled)
X_pca = pca.transform(X_scaled[0].reshape(1, -1))
plt.plot(pca.inverse_transform(X_pca)[0])
plt.plot(X_scaled[0])
plt.show()

np.random.seed(42)
kmeans = KMeans(n_clusters =5,random_state=0, n_init="auto").fit(X_projected)
kmeans.labels_


homogeneity_completeness_v_measure(Y,kmeans.labels_)

pca2 = PCA(n_components  = 2) #PCA en 2D
pca2.fit(X_scaled)
PCA(n_components = 2)
X_pca2 = pca2.transform(X_scaled)
kmeans = KMeans(n_clusters =5,random_state=0, n_init="auto").fit(X_pca2)
homogeneity_completeness_v_measure(Y,kmeans.labels_)


plt.scatter(X_pca2[:, 0], X_pca2[:, 1], c =Y, marker = 'o', alpha=0.5, edgecolor='black') #[:,0] = Toute la colonne 0
plt.title("Exact ECG label")
plt.show()
plt.scatter(X_pca2[:, 0], X_pca2[:, 1], c = kmeans.labels_, marker = 'o', alpha=0.5, edgecolor='black')
plt.title("Exact ECG label")
plt.show()


X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.30, random_state=42)

clf = LogisticRegression(random_state=42, max_iter = 10000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("score:",clf.score(X_test, y_test))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot()
plt.show()

clf2 = GaussianNB()
clf2.fit(X_train, y_train)
y_pred = clf2.predict(X_test)
print("score:",clf2.score(X_test, y_test))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf2.classes_)
disp.plot()
plt.show()



clf3 = SVC(random_state=42)
clf3.fit(X_train, y_train)
y_pred = clf3.predict(X_test)
print("score avec kernel 'rbf': ",clf3.score(X_test, y_test))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf3.classes_)
disp.plot()
plt.title("Matrice confusion avec kernel 'rbf'")
plt.show()


clf4 = SVC(kernel = 'linear', random_state=42)
clf4.fit(X_train, y_train)
y_pred = clf4.predict(X_test)
print("score avec kernel 'linear': ",clf4.score(X_test, y_test))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf4.classes_)
disp.plot()
plt.title("Matrice confusion avec kernel 'linear'")
plt.show()










