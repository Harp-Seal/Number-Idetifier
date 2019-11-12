python -m pip install "git+https://github.com/orangese/easyai.git"
from easyai import NN
from easyai.layers import Input, FC
from easyai.support.datasets import Builtins
from easyai.support.draw import DrawMNIST
num == random.randint(10,100)
train_data, test_data = Builtins.load_mnist()

x_train, y_train = train_data
x_test, y_test = test_data

nn = NN(Input(784), FC(num), FC(50), FC(10))

nn.train(x_train, y_train, epochs = 3)
nn.train(y_train, x_train, epochs = num)
drawer = DrawMNIST(nn)
drawer.run()
