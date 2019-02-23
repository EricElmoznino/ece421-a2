import unittest
from tqdm import tqdm
import os
import tensorflow as tf

from starter import *
from plotting import *


class Part1(unittest.TestCase):

    def setUp(self):
        self.train_x, self.val_x, self.test_x, self.train_y, self.val_y, self.test_y = load_data()
        self.train_y, self.val_y, self.test_y = convert_one_hot(self.train_y, self.val_y, self.test_y)

    def test_3(self):
        metric = {'Training loss': [], 'Validation loss': [], 'Test loss': [],
                  'Training accuracy': [], 'Validation accuracy': [], 'Test accuracy': []}
        epochs = 200
        lr = 0.001
        momentum = 0.9
        net = Network(1000)
        opt = Optimizer(lr, momentum, net)
        for _ in tqdm(range(0, epochs + 1)):
            train_p = net.forward(self.train_x)
            net.backward(self.train_y, train_p)
            opt.step()
            val_p = net.forward(self.val_x)
            test_p = net.forward(self.test_x)
            metric['Training loss'].append(average_ce(self.train_y, train_p))
            metric['Validation loss'].append(average_ce(self.val_y, val_p))
            metric['Test loss'].append(average_ce(self.test_y, test_p))
            metric['Training accuracy'].append(accuracy(self.train_y, train_p))
            metric['Validation accuracy'].append(accuracy(self.val_y, val_p))
            metric['Test accuracy'].append(accuracy(self.test_y, test_p))
        for title in metric:
            line_plot(title, list(range(0, epochs + 1)),
                      [metric[title]], ['learning rate=%g, momentum=%g' % (lr, momentum)],
                      'epochs', title.split(' ')[1],
                      os.path.join('results', '1_3', title + '.png'))
        with open(os.path.join('results', '1_3', 'final_metrics_alpha=%g_momentum=%g.txt' % (lr, momentum)), 'w') as f:
            for title in metric:
                f.write('%s: %g\n' % (title, metric[title][-1]))

    def test_4(self):
        hidden_units = [100, 500, 2000]
        metrics = [{'Training loss': [], 'Validation loss': [], 'Test loss': [],
                  'Training accuracy': [], 'Validation accuracy': [], 'Test accuracy': []}
                   for _ in hidden_units]
        epochs = 200
        lr = 0.001
        momentum = 0.9
        for (hidden_unit, metric) in zip(hidden_units, metrics):
            net = Network(hidden_unit)
            opt = Optimizer(lr, momentum, net)
            for _ in tqdm(range(0, epochs + 1)):
                train_p = net.forward(self.train_x)
                net.backward(self.train_y, train_p)
                opt.step()
                val_p = net.forward(self.val_x)
                test_p = net.forward(self.test_x)
                metric['Training loss'].append(average_ce(self.train_y, train_p))
                metric['Validation loss'].append(average_ce(self.val_y, val_p))
                metric['Test loss'].append(average_ce(self.test_y, test_p))
                metric['Training accuracy'].append(accuracy(self.train_y, train_p))
                metric['Validation accuracy'].append(accuracy(self.val_y, val_p))
                metric['Test accuracy'].append(accuracy(self.test_y, test_p))
        for title in metrics[0]:
            line_plot(title, list(range(0, epochs + 1)),
                      [metric[title] for metric in metrics], ['hidden units=%g' % hidden_unit],
                      'epochs', title.split(' ')[1],
                      os.path.join('results', '1_4', title + '.png'))
        for (a, m) in zip(hidden_units, metrics):
            with open(os.path.join('results', '1_4', 'hidden_units=%g.txt' % a), 'w') as f:
                for title in m:
                    f.write('%s: %g\n' % (title, m[title][-1]))

class Part2(unittest.TestCase):

    def setUp(self):
        self.train_x, self.val_x, self.test_x, self.train_y, self.val_y, self.test_y = load_data()
        self.train_y, self.val_y, self.test_y = convert_one_hot(self.train_y, self.val_y, self.test_y)

    def test_2(self):
        metric = {'Training loss': [], 'Validation loss': [], 'Test loss': [],
                  'Training accuracy': [], 'Validation accuracy': [], 'Test accuracy': []}
        epochs = 50
        batch_size = 32
        alpha = 0.0001
        print('Training with alpha=%g, batch_size=%g' % (alpha, batch_size))
        tf.reset_default_graph()
        input, predictions, targets, loss, optimizer_op, lr, reg, keep_prob = tensorflow_net()
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        for _ in tqdm(range(0, epochs + 1)):
            metric['Training loss'].append(sess.run(loss, feed_dict={input: self.train_x, targets: self.train_y, keep_prob: 1.0, reg: 0.0, lr: alpha}))
            metric['Training accuracy'].append(accuracy(self.train_y, sess.run(predictions, feed_dict={keep_prob: 1.0, input: self.train_x})))
            metric['Validation loss'].append(sess.run(loss, feed_dict={input: self.train_x, targets: self.train_y, keep_prob: 1.0, reg: 0.0, lr: alpha}))
            metric['Validation accuracy'].append(accuracy(self.train_y, sess.run(predictions, feed_dict={keep_prob: 1.0, input: self.train_x})))
            metric['Test loss'].append(sess.run(loss, feed_dict={input: self.train_x, targets: self.train_y,keep_prob: 1.0, reg: 0.0,  lr: alpha}))
            metric['Test accuracy'].append(accuracy(self.train_y, sess.run(predictions, feed_dict={keep_prob: 1.0, input: self.train_x})))
            train_x, train_y = unison_shuffled_copies(self.train_x, self.train_y)
            for batch in range(0, self.train_x.shape[0] // batch_size):
                _ = sess.run(optimizer_op, feed_dict={input: train_x[batch:batch + batch_size],
                                                     targets: train_y[batch:batch + batch_size],
                                                     reg: 0.0, lr: alpha, keep_prob: 1})
        for title in metric:
            line_plot(title, list(range(0, epochs + 1)),
                      [metric[title]], ['learning rate = %g' % alpha],
                      'epochs', title.split(' ')[1],
                      os.path.join('results', '2_2', title + '.png'))
        with open(os.path.join('results', '2_2', 'final_metrics_alpha=%g.txt' % alpha), 'w') as f:
            for title in metric:
                f.write('%s: %g\n' % (title, metric[title][-1]))

    def test_3(self):
        regs = [0.01, 0.1, 0.5]
        metrics = [{'Training loss': [], 'Validation loss': [], 'Test loss': [],
                  'Training accuracy': [], 'Validation accuracy': [], 'Test accuracy': []} for _ in regs]
        epochs = 50
        batch_size = 32
        alpha = 0.0001
        for (r, metric) in zip(regs, metrics):
            print('Training with regularizer=%g' % r)
            tf.reset_default_graph()
            input, predictions, targets, loss, optimizer_op, lr, reg, keep_prob = tensorflow_net()
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
            for _ in tqdm(range(0, epochs + 1)):
                metric['Training loss'].append(sess.run(loss, feed_dict={input: self.train_x, targets: self.train_y, keep_prob: 1.0, reg: r, lr: alpha}))
                metric['Training accuracy'].append(accuracy(self.train_y, sess.run(predictions, feed_dict={keep_prob: 1.0, input: self.train_x})))
                metric['Validation loss'].append(sess.run(loss, feed_dict={input: self.train_x, targets: self.train_y, keep_prob: 1.0, reg: r, lr: alpha}))
                metric['Validation accuracy'].append(accuracy(self.train_y, sess.run(predictions, feed_dict={keep_prob: 1.0, input: self.train_x})))
                metric['Test loss'].append(sess.run(loss, feed_dict={input: self.train_x, targets: self.train_y,keep_prob: 1.0, reg: r,  lr: alpha}))
                metric['Test accuracy'].append(accuracy(self.train_y, sess.run(predictions, feed_dict={keep_prob: 1.0, input: self.train_x})))
                train_x, train_y = unison_shuffled_copies(self.train_x, self.train_y)
                for batch in range(0, self.train_x.shape[0] // batch_size):
                    _ = sess.run(optimizer_op, feed_dict={input: train_x[batch:batch + batch_size],
                                                         targets: train_y[batch:batch + batch_size],
                                                         reg: r, lr: alpha, keep_prob: 1.0})
        for title in metrics[0]:
            line_plot(title, list(range(0, epochs + 1)),
                      [m[title] for m in metrics], ['regularizer = %g' % r for r in regs],
                      'epochs', title.split(' ')[1],
                      os.path.join('results', '2_3', title + '.png'))
        for r, m in zip(regs, metrics):
            with open(os.path.join('results', '2_3', 'final_metrics_reg=%g.txt' % r), 'w') as f:
                for title in m:
                    f.write('%s: %g\n' % (title, metric[title][-1]))

    def test_4(self):
        dropouts = [0.9, 0.75, 0.5]
        metrics = [{'Training loss': [], 'Validation loss': [], 'Test loss': [],
                    'Training accuracy': [], 'Validation accuracy': [], 'Test accuracy': []} for _ in dropouts]
        epochs = 1
        batch_size = 32
        alpha = 0.0001
        for (d, metric) in zip(dropouts, metrics):
            print('Training with dropout=%g' % d)
            tf.reset_default_graph()
            input, predictions, targets, loss, optimizer_op, lr, reg, keep_prob = tensorflow_net()
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
            for _ in tqdm(range(0, epochs + 1)):
                metric['Training loss'].append(sess.run(loss, feed_dict={input: self.train_x, targets: self.train_y,
                                                                         keep_prob: d, reg: 0.0, lr: alpha}))
                metric['Training accuracy'].append(
                    accuracy(self.train_y, sess.run(predictions, feed_dict={keep_prob: d, input: self.train_x})))
                metric['Validation loss'].append(sess.run(loss, feed_dict={input: self.train_x, targets: self.train_y,
                                                                           keep_prob: d, reg: 0.0, lr: alpha}))
                metric['Validation accuracy'].append(
                    accuracy(self.train_y, sess.run(predictions, feed_dict={keep_prob: d, input: self.train_x})))
                metric['Test loss'].append(sess.run(loss, feed_dict={input: self.train_x, targets: self.train_y,
                                                                     keep_prob: d, reg: 0.0, lr: alpha}))
                metric['Test accuracy'].append(
                    accuracy(self.train_y, sess.run(predictions, feed_dict={keep_prob: d, input: self.train_x})))
                train_x, train_y = unison_shuffled_copies(self.train_x, self.train_y)
                for batch in range(0, self.train_x.shape[0] // batch_size):
                    _ = sess.run(optimizer_op, feed_dict={input: train_x[batch:batch + batch_size],
                                                          targets: train_y[batch:batch + batch_size],
                                                          reg: 0.0, lr: alpha, keep_prob: d})
        for title in metrics[0]:
            line_plot(title, list(range(0, epochs + 1)),
                      [m[title] for m in metrics], ['dropout probability = %g' % d for d in dropouts],
                      'epochs', title.split(' ')[1],
                      os.path.join('results', '2_4', title + '.png'))
        for d, m in zip(dropouts, metrics):
            with open(os.path.join('results', '2_4', 'final_metrics_dropout=%g.txt' % d), 'w') as f:
                for title in m:
                    f.write('%s: %g\n' % (title, metric[title][-1]))

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]
