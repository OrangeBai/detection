from tensorflow.keras.optimizers.schedules import *
from utils.lr_schedule import *
import time
from queue import Queue
from  multiprocessing import Queue
from threading import Thread


class BaseModel:
    """
    Base Model
    """

    def __init__(self):
        """
        Initialization method
        """
        self.model = None  # Model
        self.result = None  # Training Result
        self.reset_args = None  # Arguments for reset the model
        self._init_rate = 0.1  # Initial learning rate
        self._lr_schedule = static_learning_rate  # Learning rate schedule
        self._loss = None  # Loss

    def build_model(self, *args, **kwargs):
        self.reset_args = [args, kwargs]

    def compile(self, optimizer, loss, metrics=None, lr_schedule=static_learning_rate, init_rate=0.1):
        """
        Compile method
        :param optimizer: Model optimizer
        :param loss: Model loss
        :param metrics: Metrics
        :param lr_schedule: Learning rate schedule function
        :param init_rate: initial learning rate
        :return:
        """
        self._init_rate = init_rate
        self._lr_schedule = lr_schedule
        self._loss = loss

        # set learning rate
        learning_rate_fn = InverseTimeDecay(init_rate, 1, decay_rate=1e-5)
        optimizer.learning_rate = learning_rate_fn

        self.model.compile(optimizer=optimizer,
                           loss=loss,
                           metrics=metrics)

    def train_epoch(self, batch_num, train_gen):
        start_time = time.time()
        metrics_name = self.model.metrics_names
        metrics_num = len(metrics_name)

        train_res = np.zeros((batch_num, metrics_num))
        # for j in range(batch_num):
        #     x_train, y_train = next(train_gen)
        #     train_res[j, :] = self.model.train_on_batch(np.array(x_train), np.array(y_train))

        q = Queue(10)

        producer = Thread(target=self.producer, args=(q, batch_num, train_gen))
        consumer = Thread(target=self.consumer, args=(q, batch_num, train_res))
        producer.start()
        consumer.start()
        producer.join()
        consumer.join()

        train_res = train_res.mean(axis=0)
        print('Val time: {0}'.format(time.time() - start_time))
        log = 'Train - '
        for idx, name in enumerate(metrics_name):
            log += '{0}: {1} - '.format(name, train_res[idx])
        print(log)

        return train_res, metrics_name

    def update_lr(self, epoch_num, cur_epoch):
        new_lr = self._lr_schedule(epoch_num, cur_epoch, self._init_rate)
        if new_lr:
            learning_rate = InverseTimeDecay(new_lr, 1, decay_rate=1e-4)
            self.model.optimizer.learning_rate = learning_rate
            print(new_lr)

    def evaluate_on_generator(self, val_gen):
        pre_res = []
        gt_res = []
        while True:
            try:
                x_val, y_val = next(val_gen)
                res = self.model.predict_on_batch(np.array(x_val))
                for cur_res, cur_gt in zip(res, y_val):
                    pre_res.append(cur_res)
                    gt_res.append(cur_gt)
            except StopIteration:
                break
        return pre_res, gt_res

    def multiple_training(self, exp_num):
        pass

    def reset_model(self):
        self.build_model()

    def save_model(self, path):
        self.model.save_weights(path)

    @staticmethod
    def producer(q, batch_num, gen):
        for j in range(batch_num):
            x = next(gen)
            q.put(x)
        q.put([None, None])

    def consumer(self, q, batch_num, train_res):
        for j in range(batch_num):
            x_train, y_train = q.get()
            if x_train is None:
                break
            train_res[j, :] = self.model.train_on_batch(np.array(x_train), np.array(y_train))
