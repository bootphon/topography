import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_iris
from topography.som import SOM

iris = load_iris()
iris_data = iris.data[:, :2]
iris_label = iris.target

som = SOM(m=3, n=1).fit(iris_data)
predictions = som.predict(iris_data)

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(5, 7))
x = iris_data[:, 0]
y = iris_data[:, 1]
colors = ['red', 'green', 'blue']

ax[0].scatter(x, y, c=iris_label, cmap=ListedColormap(colors))
ax[0].title.set_text('Actual Classes')
ax[1].scatter(x, y, c=predictions, cmap=ListedColormap(colors))
ax[1].title.set_text('SOM Predictions')
plt.show()
