import chainer
import chainer.links as L
import chainer.functions as F


class SimplePredictor(chainer.Chain):
    """
    推論器のクラスです。
    デフォルトでは全結合層３つの間に活性化関数reluを入れて構成しています。
    """
    def __init__(self, unit_num1, unit_num2, output_num):
        super(SimplePredictor, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, unit_num1)
            self.l2 = L.Linear(None, unit_num2)
            self.l3 = L.Linear(None, output_num)

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        y = self.l3(h2)
        return y

class SimpleRegression(chainer.Chain):
    """
    ニューラルネットワークを進めていくためのクラスです。
    デフォルトではSimplePredictorを推論器に使い
    回帰的に実数の値を返す関数を学習するようになっています。
    """
    def __init__(self, predictor):
        super(SimpleRegression, self).__init__()
        with self.init_scope():
            self.predictor = predictor

    def __call__(self, x, t):
        y = self.predictor(x)
        loss =  F.mean_squared_error(y, t)
        r2_score = F.r2_score(y, t)
        chainer.report({'loss': loss, 'r2_score': r2_score}, self)
        return loss

