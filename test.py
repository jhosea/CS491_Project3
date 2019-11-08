import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt


def plot_decision_boundary(pred_fun, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    h = 0.01

    # Generate a grid of points with a distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Predict the function value for the whole grid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)


np.random.seed(0)
X, y = make_moons(200, noise=0.20)
plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)

# plt.figure(figsize=(16, 32))
# hidden_layer_dimensions = [1, 2, 3, 4]
# for i, nnh_dim in enumerate(hiddenlayerdimensions):
#     plt.subplot(5, 2, i+1)
#     plt.title(’HiddenLayerSize % d’ % nn_hdim)
#     model = buildmodel(X, y, nn_hdim)
#     plot_decision_boundary(lambdax: predict(model, x), X, y)
# plt.show()
