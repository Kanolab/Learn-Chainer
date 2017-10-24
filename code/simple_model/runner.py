import simple
import numpy

"""
深層学習のコード
Xが入力で、yが出力です。
デフォルトでは関数 f(x) = 2*x を学習するようにしています。
"""
if __name__ == "__main__":
    X = numpy.random.rand(10000) * 100
    X = X.astype(numpy.float32)
    X = numpy.expand_dims(X, axis=1)

    y = 2 * X
    simple.run(X, y, 50.0, 100.0)
