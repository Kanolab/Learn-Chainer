import simple
import numpy
from random import random

def main():
    """
    ニューラルネットワークのコード
    Xが入力で、yが出力です。
    デフォルトでは関数 f(x) = 2*x を学習するようにしています。
    """
    X = []
    for i in range(10000):
        X.append(random() * 100)
    X = numpy.array(X)
    X = X.astype(numpy.float32)
    X = numpy.expand_dims(X, axis=1)

    y = 2 * X
    simple.run(X, y, 1.0, 2.0)

if __name__ == "__main__":
    main()