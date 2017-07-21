# coding=utf-8
class TrainingConfig(object):
    decay_step = 15000
    decay_rate = 0.95
    epoches = 50000
    evaluate_every = 100
    checkpoint_every = 100

class ModelConfig(object):
    conv_layers = [[256, 7, 3],
                   [256, 7, 3],
                   [256, 3, None],
                   [256, 3, None],
                   [256, 3, None],
                   [256, 3, 3]]

    fc_layers = [1024, 1024]
    dropout_keep_prob = 0.9
    learning_rate = 0.001

class Config(object):
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
    alphabet_size = len(alphabet)
    l0 = 1014
    batch_size = 128
    nums_classes = 4
    example_nums = 120000

    train_data_source = 'data/ag_news_csv/train.csv'
    dev_data_source = 'data/ag_news_csv/test.csv'

    training = TrainingConfig()

    model = ModelConfig()


config = Config()
