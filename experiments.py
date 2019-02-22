import unittest
from tqdm import tqdm
import os
import tensorflow as tf

from starter import *
from plotting import *


class Part1(unittest.TestCase):

    def setUp(self):
        self.train_x, self.val_x, self.test_x, self.train_y, self.val_y, self.test_y = load_data()
        self.train_y, self.val_y, self.text_y = convert_one_hot(self.train_y, self.val_y, self.test_y)

    def test_3(self):
        metrics = {'Training loss': [], 'Validation loss': [], 'Test loss': [],
                    'Training accuracy': [], 'Validation accuracy': [], 'Test accuracy': []}
        epochs = 200
        save_freq = 100
        for metric in zip(metrics):
            Network.__init__(1000)
            for _ in tqdm(range(0, epochs + 1,)):
                metric['Training loss'].append(average_ce(self.train_y, self.train_x,))
                metric['Validation loss'].append(average_ce(self.val_y, self.val_x))
                metric['Test loss'].append(average_ce(w, b, self.test_y, self.test_x))
                metric['Training accuracy'].append(average_ce(self.train_y, self.train_x))
                metric['Validation accuracy'].append(average_ce(self.val_y, self.val_x))
                metric['Test accuracy'].append(average_ce( self.test_y, self.test_x))
                w, b = grad_ce(w, b, self.train_x, self.train_y, alpha, save_freq, 0.0, 1e-7, 'MSE')
        for title in metrics[0]:
            line_plot(title, list(range(0, epochs + 1, save_freq)),
                      [m[title] for m in metrics], ['learning rate = %g' % a for a in alphas],
                      'epochs', title.split(' ')[1],
                      os.path.join('results', '1_3', title + '.png'))
        for a, m in zip(alphas, metrics):
            with open(os.path.join('results', '1_3', 'final_metrics_alpha=%g.txt' % a), 'w') as f:
                for title in m:
                    f.write('%s: %g\n' % (title, m[title][-1]))

    def test_4(self):
        regs = [0.001, 0.1, 0.5]
        metrics = [{'Training loss': [], 'Validation loss': [], 'Test loss': [],
                    'Training accuracy': [], 'Validation accuracy': [], 'Test accuracy': []}
                   for _ in regs]
        epochs = 5000
        save_freq = 100
        for reg, metric in zip(regs, metrics):
            print('Training with reg=%g' % reg)
            w, b = initialize()
            for _ in tqdm(range(0, epochs + 1, save_freq)):
                metric['Training loss'].append(mse(w, b, self.train_x, self.train_y, reg))
                metric['Validation loss'].append(mse(w, b, self.val_x, self.val_y, reg))
                metric['Test loss'].append(mse(w, b, self.test_x, self.test_y, reg))
                metric['Training accuracy'].append(accuracy(w, b, self.train_x, self.train_y))
                metric['Validation accuracy'].append(accuracy(w, b, self.val_x, self.val_y))
                metric['Test accuracy'].append(accuracy(w, b, self.test_x, self.test_y))
                w, b = grad_descent(w, b, self.train_x, self.train_y, 0.005, save_freq, reg, 1e-7, 'MSE')
        for title in metrics[0]:
            line_plot(title, list(range(0, epochs + 1, save_freq)),
                      [m[title] for m in metrics], ['regularization = %g' % a for a in regs],
                      'epochs', title.split(' ')[1],
                      os.path.join('results', '1_4', title + '.png'))
        for a, m in zip(regs, metrics):
            with open(os.path.join('results', '1_4', 'final_metrics_reg=%g.txt' % a), 'w') as f:
                for title in m:
                    f.write('%s: %g\n' % (title, m[title][-1]))

    def test_5(self):
        def closed_form_optimal(x, y):
            x = np.concatenate([np.ones((x.shape[0], 1), dtype=x.dtype), x], axis=1)
            w = np.matmul(np.matmul(np.linalg.inv(np.matmul(x.transpose(), x)), x.transpose()), y)
            w, b = w[1:], w[:1]
            return w, b

        w, b = closed_form_optimal(self.train_x, self.train_y)
        metrics = {'Training loss': mse(w, b, self.train_x, self.train_y, 0.0),
                   'Validation loss': mse(w, b, self.val_x, self.val_y, 0.0),
                   'Test loss': mse(w, b, self.test_x, self.test_y, 0.0),
                   'Training accuracy': accuracy(w, b, self.train_x, self.train_y),
                   'Validation accuracy': accuracy(w, b, self.val_x, self.val_y),
                   'Test accuracy': accuracy(w, b, self.test_x, self.test_y)}
        with open(os.path.join('results', '1_5', 'final_metrics.txt'), 'w') as f:
            for title in metrics:
                f.write('%s: %g\n' % (title, metrics[title]))


class Part2(unittest.TestCase):

    def setUp(self):
        self.train_x, self.val_x, self.test_x, self.train_y, self.val_y, self.test_y = load_data()

    def test_2(self):
        alpha = 0.005
        metric = {'Training loss': [], 'Validation loss': [], 'Test loss': [],
                  'Training accuracy': [], 'Validation accuracy': [], 'Test accuracy': []}
        epochs = 5000
        save_freq = 100
        print('Training with alpha=%g' % alpha)
        w, b = initialize()
        for _ in tqdm(range(0, epochs + 1, save_freq)):
            metric['Training loss'].append(cross_entropy_loss(w, b, self.train_x, self.train_y, 0.1))
            metric['Validation loss'].append(cross_entropy_loss(w, b, self.val_x, self.val_y, 0.1))
            metric['Test loss'].append(cross_entropy_loss(w, b, self.test_x, self.test_y, 0.1))
            metric['Training accuracy'].append(accuracy(w, b, self.train_x, self.train_y, ce=True))
            metric['Validation accuracy'].append(accuracy(w, b, self.val_x, self.val_y, ce=True))
            metric['Test accuracy'].append(accuracy(w, b, self.test_x, self.test_y, ce=True))
            w, b = grad_descent(w, b, self.train_x, self.train_y, alpha, save_freq, 0.1, 1e-7, 'CE')
        for title in metric:
            line_plot(title, list(range(0, epochs + 1, save_freq)),
                      [metric[title]], ['learning rate = %g' % alpha],
                      'epochs', title.split(' ')[1],
                      os.path.join('results', '2_2', title + '.png'))
        with open(os.path.join('results', '2_2', 'final_metrics=%g.txt' % alpha), 'w') as f:
            for title in metric:
                f.write('%s: %g\n' % (title, metric[title][-1]))

    def test_3(self):
        alpha = 0.005
        metric = {'Training loss': [], 'Validation loss': [], 'Test loss': [],
                  'Training accuracy': [], 'Validation accuracy': [], 'Test accuracy': []}
        epochs = 5000
        save_freq = 100
        print('Training with alpha=%g' % alpha)
        w, b = initialize()
        for _ in tqdm(range(0, epochs + 1, save_freq)):
            metric['Training loss'].append(cross_entropy_loss(w, b, self.train_x, self.train_y, 0.0))
            metric['Validation loss'].append(cross_entropy_loss(w, b, self.val_x, self.val_y, 0.0))
            metric['Test loss'].append(cross_entropy_loss(w, b, self.test_x, self.test_y, 0.0))
            metric['Training accuracy'].append(accuracy(w, b, self.train_x, self.train_y, ce=True))
            metric['Validation accuracy'].append(accuracy(w, b, self.val_x, self.val_y, ce=True))
            metric['Test accuracy'].append(accuracy(w, b, self.test_x, self.test_y, ce=True))
            w, b = grad_descent(w, b, self.train_x, self.train_y, alpha, save_freq, 0.0, 1e-7, 'CE')
        for title in metric:
            line_plot(title, list(range(0, epochs + 1, save_freq)),
                      [metric[title]], ['learning rate = %g' % alpha],
                      'epochs', title.split(' ')[1],
                      os.path.join('results', '2_3', title + '.png'))
        with open(os.path.join('results', '2_3', 'final_metrics=%g.txt' % alpha), 'w') as f:
            for title in metric:
                f.write('%s: %g\n' % (title, metric[title][-1]))


class Part3(unittest.TestCase):

    def setUp(self):
        self.train_x, self.val_x, self.test_x, self.train_y, self.val_y, self.test_y = load_data()

    def test_2(self):
        alpha = 0.001
        metric = {'Training loss': [], 'Validation loss': [], 'Test loss': [],
                  'Training accuracy': [], 'Validation accuracy': [], 'Test accuracy': []}
        epochs = 700
        batch_size = 500
        print('Training with alpha=%g' % alpha)
        x, y_hat, y, w, b, loss, optimize_op, reg = build_graph(learning_rate=alpha, loss_type='MSE')
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        for _ in tqdm(range(0, epochs + 1)):
            metric['Training loss'].append(sess.run(loss, feed_dict={x: self.train_x, y: self.train_y, reg: 0.0}))
            metric['Validation loss'].append(sess.run(loss, feed_dict={x: self.val_x, y: self.val_y, reg: 0.0}))
            metric['Test loss'].append(sess.run(loss, feed_dict={x: self.test_x, y: self.test_y, reg: 0.0}))
            metric['Training accuracy'].append(accuracy(sess.run(w), sess.run(b), self.train_x, self.train_y))
            metric['Validation accuracy'].append(accuracy(sess.run(w), sess.run(b), self.val_x, self.val_y))
            metric['Test accuracy'].append(accuracy(sess.run(w), sess.run(b), self.test_x, self.test_y))
            train_x, train_y = unison_shuffled_copies(self.train_x, self.train_y)
            for batch in range(0, self.train_x.shape[0] // batch_size):
                _ = sess.run(optimize_op, feed_dict={x: train_x[batch:batch + batch_size],
                                                     y: train_y[batch:batch + batch_size],
                                                     reg: 0.0})
        for title in metric:
            line_plot(title, list(range(0, epochs + 1)),
                      [metric[title]], ['learning rate = %g' % alpha],
                      'epochs', title.split(' ')[1],
                      os.path.join('results', '3_2', title + '.png'))
        with open(os.path.join('results', '3_2', 'final_metrics=%g.txt' % alpha), 'w') as f:
            for title in metric:
                f.write('%s: %g\n' % (title, metric[title][-1]))

    def test_3(self):
        batch_sizes = [100, 700, 1750]
        metrics = [{'Training loss': [], 'Validation loss': [], 'Test loss': [],
                    'Training accuracy': [], 'Validation accuracy': [], 'Test accuracy': []}
                   for _ in batch_sizes]
        epochs = 700
        for batch_size, metric in zip(batch_sizes, metrics):
            print('Training with batch_size=%d' % batch_size)
            tf.reset_default_graph()
            x, y_hat, y, w, b, loss, optimize_op, reg = build_graph(learning_rate=0.001, loss_type='MSE')
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
            for _ in tqdm(range(0, epochs + 1)):
                metric['Training loss'].append(sess.run(loss, feed_dict={x: self.train_x, y: self.train_y, reg: 0.0}))
                metric['Validation loss'].append(sess.run(loss, feed_dict={x: self.val_x, y: self.val_y, reg: 0.0}))
                metric['Test loss'].append(sess.run(loss, feed_dict={x: self.test_x, y: self.test_y, reg: 0.0}))
                metric['Training accuracy'].append(accuracy(sess.run(w), sess.run(b), self.train_x, self.train_y))
                metric['Validation accuracy'].append(accuracy(sess.run(w), sess.run(b), self.val_x, self.val_y))
                metric['Test accuracy'].append(accuracy(sess.run(w), sess.run(b), self.test_x, self.test_y))
                train_x, train_y = unison_shuffled_copies(self.train_x, self.train_y)
                for batch in range(0, self.train_x.shape[0] // batch_size):
                    _ = sess.run(optimize_op, feed_dict={x: train_x[batch:batch + batch_size],
                                                         y: train_y[batch:batch + batch_size],
                                                         reg: 0.0})
        for title in metrics[0]:
            line_plot(title, list(range(0, epochs + 1)),
                      [m[title] for m in metrics], ['batch size = %d' % b for b in batch_sizes],
                      'epochs', title.split(' ')[1],
                      os.path.join('results', '3_3', title + '.png'))
        for b, m in zip(batch_sizes, metrics):
            with open(os.path.join('results', '3_3', 'final_metrics_batch_size=%d.txt' % b), 'w') as f:
                for title in m:
                    f.write('%s: %g\n' % (title, m[title][-1]))

    def test_4_1(self):
        beta1s = [0.95, 0.99]
        metrics = [{'Training loss': [], 'Validation loss': [], 'Test loss': [],
                    'Training accuracy': [], 'Validation accuracy': [], 'Test accuracy': []}
                   for _ in beta1s]
        epochs = 700
        batch_size = 500
        for b1, metric in zip(beta1s, metrics):
            print('Training with beta1=%g' % b1)
            tf.reset_default_graph()
            x, y_hat, y, w, b, loss, optimize_op, reg = build_graph(learning_rate=0.001, beta1=b1, loss_type='MSE')
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
            for _ in tqdm(range(0, epochs + 1)):
                metric['Training loss'].append(sess.run(loss, feed_dict={x: self.train_x, y: self.train_y, reg: 0.0}))
                metric['Validation loss'].append(sess.run(loss, feed_dict={x: self.val_x, y: self.val_y, reg: 0.0}))
                metric['Test loss'].append(sess.run(loss, feed_dict={x: self.test_x, y: self.test_y, reg: 0.0}))
                metric['Training accuracy'].append(accuracy(sess.run(w), sess.run(b), self.train_x, self.train_y))
                metric['Validation accuracy'].append(accuracy(sess.run(w), sess.run(b), self.val_x, self.val_y))
                metric['Test accuracy'].append(accuracy(sess.run(w), sess.run(b), self.test_x, self.test_y))
                train_x, train_y = unison_shuffled_copies(self.train_x, self.train_y)
                for batch in range(0, self.train_x.shape[0] // batch_size):
                    _ = sess.run(optimize_op, feed_dict={x: train_x[batch:batch + batch_size],
                                                         y: train_y[batch:batch + batch_size],
                                                         reg: 0.0})
        for title in metrics[0]:
            line_plot(title, list(range(0, epochs + 1)),
                      [m[title] for m in metrics], ['beta1=%g' % b1 for b1 in beta1s],
                      'epochs', title.split(' ')[1],
                      os.path.join('results', '3_4_1', title + '.png'))
        for b1, m in zip(beta1s, metrics):
            with open(os.path.join('results', '3_4_1',
                                   'final_metrics_beta1=%g.txt' % b1), 'w') as f:
                for title in m:
                    f.write('%s: %g\n' % (title, m[title][-1]))

    def test_4_2(self):
        beta2s = [0.99, 0.9999]
        metrics = [{'Training loss': [], 'Validation loss': [], 'Test loss': [],
                    'Training accuracy': [], 'Validation accuracy': [], 'Test accuracy': []}
                   for _ in beta2s]
        epochs = 700
        batch_size = 500
        for b2, metric in zip(beta2s, metrics):
            print('Training with beta2=%g' % b2)
            tf.reset_default_graph()
            x, y_hat, y, w, b, loss, optimize_op, reg = build_graph(learning_rate=0.001, beta2=b2, loss_type='MSE')
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
            for _ in tqdm(range(0, epochs + 1)):
                metric['Training loss'].append(sess.run(loss, feed_dict={x: self.train_x, y: self.train_y, reg: 0.0}))
                metric['Validation loss'].append(sess.run(loss, feed_dict={x: self.val_x, y: self.val_y, reg: 0.0}))
                metric['Test loss'].append(sess.run(loss, feed_dict={x: self.test_x, y: self.test_y, reg: 0.0}))
                metric['Training accuracy'].append(accuracy(sess.run(w), sess.run(b), self.train_x, self.train_y))
                metric['Validation accuracy'].append(accuracy(sess.run(w), sess.run(b), self.val_x, self.val_y))
                metric['Test accuracy'].append(accuracy(sess.run(w), sess.run(b), self.test_x, self.test_y))
                train_x, train_y = unison_shuffled_copies(self.train_x, self.train_y)
                for batch in range(0, self.train_x.shape[0] // batch_size):
                    _ = sess.run(optimize_op, feed_dict={x: train_x[batch:batch + batch_size],
                                                         y: train_y[batch:batch + batch_size],
                                                         reg: 0.0})
        for title in metrics[0]:
            line_plot(title, list(range(0, epochs + 1)),
                      [m[title] for m in metrics], ['beta2=%g' % b2 for b2 in beta2s],
                      'epochs', title.split(' ')[1],
                      os.path.join('results', '3_4_2', title + '.png'))
        for b2, m in zip(beta2s, metrics):
            with open(os.path.join('results', '3_4_2',
                                   'final_metrics_beta2=%g.txt' % b2), 'w') as f:
                for title in m:
                    f.write('%s: %g\n' % (title, m[title][-1]))

    def test_4_3(self):
        epsilons = [1e-9, 1e-4]
        metrics = [{'Training loss': [], 'Validation loss': [], 'Test loss': [],
                    'Training accuracy': [], 'Validation accuracy': [], 'Test accuracy': []}
                   for _ in epsilons]
        epochs = 700
        batch_size = 500
        for e, metric in zip(epsilons, metrics):
            print('Training with epsilon=%g' % e)
            tf.reset_default_graph()
            x, y_hat, y, w, b, loss, optimize_op, reg = build_graph(learning_rate=0.001, epsilon=e, loss_type='MSE')
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
            for _ in tqdm(range(0, epochs + 1)):
                metric['Training loss'].append(sess.run(loss, feed_dict={x: self.train_x, y: self.train_y, reg: 0.0}))
                metric['Validation loss'].append(sess.run(loss, feed_dict={x: self.val_x, y: self.val_y, reg: 0.0}))
                metric['Test loss'].append(sess.run(loss, feed_dict={x: self.test_x, y: self.test_y, reg: 0.0}))
                metric['Training accuracy'].append(accuracy(sess.run(w), sess.run(b), self.train_x, self.train_y))
                metric['Validation accuracy'].append(accuracy(sess.run(w), sess.run(b), self.val_x, self.val_y))
                metric['Test accuracy'].append(accuracy(sess.run(w), sess.run(b), self.test_x, self.test_y))
                train_x, train_y = unison_shuffled_copies(self.train_x, self.train_y)
                for batch in range(0, self.train_x.shape[0] // batch_size):
                    _ = sess.run(optimize_op, feed_dict={x: train_x[batch:batch + batch_size],
                                                         y: train_y[batch:batch + batch_size],
                                                         reg: 0.0})
        for title in metrics[0]:
            line_plot(title, list(range(0, epochs + 1)),
                      [m[title] for m in metrics], ['epsilon=%g' % e for e in epsilons],
                      'epochs', title.split(' ')[1],
                      os.path.join('results', '3_4_3', title + '.png'))
        for e, m in zip(epsilons, metrics):
            with open(os.path.join('results', '3_4_3',
                                   'final_metrics_epsilon=%g.txt' % e), 'w') as f:
                for title in m:
                    f.write('%s: %g\n' % (title, m[title][-1]))

    def test_5_1(self):
        alpha = 0.001
        metric = {'Training loss': [], 'Validation loss': [], 'Test loss': [],
                  'Training accuracy': [], 'Validation accuracy': [], 'Test accuracy': []}
        epochs = 700
        batch_size = 500
        print('Training with alpha=%g' % alpha)
        x, y_hat, y, w, b, loss, optimize_op, reg = build_graph(learning_rate=alpha, loss_type='CE')
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        for _ in tqdm(range(0, epochs + 1)):
            metric['Training loss'].append(sess.run(loss, feed_dict={x: self.train_x, y: self.train_y, reg: 0.0}))
            metric['Validation loss'].append(sess.run(loss, feed_dict={x: self.val_x, y: self.val_y, reg: 0.0}))
            metric['Test loss'].append(sess.run(loss, feed_dict={x: self.test_x, y: self.test_y, reg: 0.0}))
            metric['Training accuracy'].append(accuracy(sess.run(w), sess.run(b), self.train_x, self.train_y))
            metric['Validation accuracy'].append(accuracy(sess.run(w), sess.run(b), self.val_x, self.val_y))
            metric['Test accuracy'].append(accuracy(sess.run(w), sess.run(b), self.test_x, self.test_y))
            train_x, train_y = unison_shuffled_copies(self.train_x, self.train_y)
            for batch in range(0, self.train_x.shape[0] // batch_size):
                _ = sess.run(optimize_op, feed_dict={x: train_x[batch:batch + batch_size],
                                                     y: train_y[batch:batch + batch_size],
                                                     reg: 0.0})
        for title in metric:
            line_plot(title, list(range(0, epochs + 1)),
                      [metric[title]], ['learning rate = %g' % alpha],
                      'epochs', title.split(' ')[1],
                      os.path.join('results', '3_5_1', title + '.png'))
        with open(os.path.join('results', '3_5_1', 'final_metrics=%g.txt' % alpha), 'w') as f:
            for title in metric:
                f.write('%s: %g\n' % (title, metric[title][-1]))

    def test_5_2(self):
        beta1s = [0.95, 0.99]
        metrics = [{'Training loss': [], 'Validation loss': [], 'Test loss': [],
                    'Training accuracy': [], 'Validation accuracy': [], 'Test accuracy': []}
                   for _ in beta1s]
        epochs = 700
        batch_size = 500
        for b1, metric in zip(beta1s, metrics):
            print('Training with beta1=%g' % b1)
            tf.reset_default_graph()
            x, y_hat, y, w, b, loss, optimize_op, reg = build_graph(learning_rate=0.001, beta1=b1, loss_type='CE')
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
            for _ in tqdm(range(0, epochs + 1)):
                metric['Training loss'].append(sess.run(loss, feed_dict={x: self.train_x, y: self.train_y, reg: 0.0}))
                metric['Validation loss'].append(sess.run(loss, feed_dict={x: self.val_x, y: self.val_y, reg: 0.0}))
                metric['Test loss'].append(sess.run(loss, feed_dict={x: self.test_x, y: self.test_y, reg: 0.0}))
                metric['Training accuracy'].append(accuracy(sess.run(w), sess.run(b), self.train_x, self.train_y))
                metric['Validation accuracy'].append(accuracy(sess.run(w), sess.run(b), self.val_x, self.val_y))
                metric['Test accuracy'].append(accuracy(sess.run(w), sess.run(b), self.test_x, self.test_y))
                train_x, train_y = unison_shuffled_copies(self.train_x, self.train_y)
                for batch in range(0, self.train_x.shape[0] // batch_size):
                    _ = sess.run(optimize_op, feed_dict={x: train_x[batch:batch + batch_size],
                                                         y: train_y[batch:batch + batch_size],
                                                         reg: 0.0})
        for title in metrics[0]:
            line_plot(title, list(range(0, epochs + 1)),
                      [m[title] for m in metrics], ['beta1=%g' % b1 for b1 in beta1s],
                      'epochs', title.split(' ')[1],
                      os.path.join('results', '3_5_2', title + '.png'))
        for b1, m in zip(beta1s, metrics):
            with open(os.path.join('results', '3_5_2',
                                   'final_metrics_beta1=%g.txt' % b1), 'w') as f:
                for title in m:
                    f.write('%s: %g\n' % (title, m[title][-1]))

    def test_5_3(self):
        beta2s = [0.99, 0.9999]
        metrics = [{'Training loss': [], 'Validation loss': [], 'Test loss': [],
                    'Training accuracy': [], 'Validation accuracy': [], 'Test accuracy': []}
                   for _ in beta2s]
        epochs = 700
        batch_size = 500
        for b2, metric in zip(beta2s, metrics):
            print('Training with beta2=%g' % b2)
            tf.reset_default_graph()
            x, y_hat, y, w, b, loss, optimize_op, reg = build_graph(learning_rate=0.001, beta2=b2, loss_type='CE')
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
            for _ in tqdm(range(0, epochs + 1)):
                metric['Training loss'].append(sess.run(loss, feed_dict={x: self.train_x, y: self.train_y, reg: 0.0}))
                metric['Validation loss'].append(sess.run(loss, feed_dict={x: self.val_x, y: self.val_y, reg: 0.0}))
                metric['Test loss'].append(sess.run(loss, feed_dict={x: self.test_x, y: self.test_y, reg: 0.0}))
                metric['Training accuracy'].append(accuracy(sess.run(w), sess.run(b), self.train_x, self.train_y))
                metric['Validation accuracy'].append(accuracy(sess.run(w), sess.run(b), self.val_x, self.val_y))
                metric['Test accuracy'].append(accuracy(sess.run(w), sess.run(b), self.test_x, self.test_y))
                train_x, train_y = unison_shuffled_copies(self.train_x, self.train_y)
                for batch in range(0, self.train_x.shape[0] // batch_size):
                    _ = sess.run(optimize_op, feed_dict={x: train_x[batch:batch + batch_size],
                                                         y: train_y[batch:batch + batch_size],
                                                         reg: 0.0})
        for title in metrics[0]:
            line_plot(title, list(range(0, epochs + 1)),
                      [m[title] for m in metrics], ['beta2=%g' % b2 for b2 in beta2s],
                      'epochs', title.split(' ')[1],
                      os.path.join('results', '3_5_3', title + '.png'))
        for b2, m in zip(beta2s, metrics):
            with open(os.path.join('results', '3_5_3',
                                   'final_metrics_beta2=%g.txt' % b2), 'w') as f:
                for title in m:
                    f.write('%s: %g\n' % (title, m[title][-1]))

    def test_5_4(self):
        epsilons = [1e-9, 1e-4]
        metrics = [{'Training loss': [], 'Validation loss': [], 'Test loss': [],
                    'Training accuracy': [], 'Validation accuracy': [], 'Test accuracy': []}
                   for _ in epsilons]
        epochs = 700
        batch_size = 500
        for e, metric in zip(epsilons, metrics):
            print('Training with epsilon=%g' % e)
            tf.reset_default_graph()
            x, y_hat, y, w, b, loss, optimize_op, reg = build_graph(learning_rate=0.001, epsilon=e, loss_type='CE')
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
            for _ in tqdm(range(0, epochs + 1)):
                metric['Training loss'].append(sess.run(loss, feed_dict={x: self.train_x, y: self.train_y, reg: 0.0}))
                metric['Validation loss'].append(sess.run(loss, feed_dict={x: self.val_x, y: self.val_y, reg: 0.0}))
                metric['Test loss'].append(sess.run(loss, feed_dict={x: self.test_x, y: self.test_y, reg: 0.0}))
                metric['Training accuracy'].append(accuracy(sess.run(w), sess.run(b), self.train_x, self.train_y))
                metric['Validation accuracy'].append(accuracy(sess.run(w), sess.run(b), self.val_x, self.val_y))
                metric['Test accuracy'].append(accuracy(sess.run(w), sess.run(b), self.test_x, self.test_y))
                train_x, train_y = unison_shuffled_copies(self.train_x, self.train_y)
                for batch in range(0, self.train_x.shape[0] // batch_size):
                    _ = sess.run(optimize_op, feed_dict={x: train_x[batch:batch + batch_size],
                                                         y: train_y[batch:batch + batch_size],
                                                         reg: 0.0})
        for title in metrics[0]:
            line_plot(title, list(range(0, epochs + 1)),
                      [m[title] for m in metrics], ['epsilon=%g' % e for e in epsilons],
                      'epochs', title.split(' ')[1],
                      os.path.join('results', '3_5_4', title + '.png'))
        for e, m in zip(epsilons, metrics):
            with open(os.path.join('results', '3_5_4',
                                   'final_metrics_epsilon=%g.txt' % e), 'w') as f:
                for title in m:
                    f.write('%s: %g\n' % (title, m[title][-1]))


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]
