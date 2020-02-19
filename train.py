import random
import losswise
import time

losswise.set_api_key('DJOOF0FUN')
session = losswise.Session(tag='my_dilated_convnet', max_iter=10,
                           params={'cnn_size': 20})
graph = session.graph('loss', kind='min')
session.set_values({ 'key5': 'eee1'})
for x in range(10):
    train_loss = 1. / (0.1 + x + 0.1 * random.random())
    test_loss = 1.5 / (0.1 + x + 0.2 * random.random())
    graph.append(x, {'train_loss': train_loss, 'test_loss': test_loss})
    time.sleep(1.)
session.done()
