import random
import losswise
import time

losswise.set_api_key('VSDITT0OC')
session = losswise.Session(tag='my_dilated_convnet', max_iter=10,
                           params={'cnn_size': 20})
graph = session.graph('loss', kind='min')
for x in range(3600):
    train_loss = 1. / (0.1 + 0.05 * x + 0.1 * random.random())
    test_loss = 1.5 / (0.1 + 0.05 * x + 0.2 * random.random())
    print (x, {'train_loss': train_loss, 'test_loss': test_loss})
    graph.append(x, {'train_loss': train_loss, 'test_loss': test_loss})
    time.sleep(1.)
    if x % 500 == 0:
        seq = session.image_sequence(x=x, name="Test")
        for img_id in range(5):
            pil_image = Image.open("image.png")
            seq.append(pil_image,
                       metrics={'accuracy': 1},
                       outputs={'name': 'Lena'},
                       image_id=str(img_id) + "_img")
session.done()