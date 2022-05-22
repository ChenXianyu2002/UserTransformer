import os

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from SearchDataLoader import DataSet, PointDataSet, SimpleDataset
from utils import load_model


def load_dataset(args):
    data_paths = {
        'taobao': 'TaoBao/taobao/raw_data',
        'tmall': 'TaoBao/tmall/raw_data',
        'alipay': 'TaoBao/alipay/raw_data',
    }
    dataset = os.path.join(args['data_dir'], data_paths[args['dataset']])
    return np.load(os.path.join(dataset, 'user_item_multi.npz'))


def main(args):
    print(args)
    user_item = load_dataset(args)
    args.update({'all_features_num': np.sum(user_item['fields'])})
    model = load_model(args)
    if args['model'] == 'PNN':
        dataset = PointDataSet(user_item, args['batch_size'])
    elif args['model'] == 'SimpleSTARec':
        dataset = SimpleDataset(user_item, args['batch_size'], args['item_num'], args['hard_search'])
    else:
        dataset = DataSet(user_item, args['batch_size'], args['item_num'], args['hard_search'])
    print("dataset", args["dataset"])
    print("hidden dim", args["hidden_dim"])
    print("lr", args["lr"])
    model_folder = "./saved_models/"
    model_path = model_folder + str(args["exp_name"]) + "_" + str(args["postfix"])
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    train_data = dataset.input_fn('train', epochs=args['num_epochs'])
    valid_data = dataset.input_fn('valid', epochs=None)
    test_data = dataset.input_fn('test', epochs=1)
    train_base_loss, train_loss, train_eval_metric_ops, train_y_, train_y, train_op = model(train_data)
    valid_base_loss, valid_loss, valid_eval_metric_ops, valid_y_, valid_y = model(valid_data, False)
    test_base_loss, test_loss, test_eval_metric_ops, test_y_, test_y = model(test_data, False)
    saver = tf.train.Saver(max_to_keep=2)
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    best_val_auc = 0
    step_up = len(dataset.user_log_id) // args['batch_size']
    print("-----------------Training Start-----------------\n")
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        if args['load_model']:
            saver.restore(sess, model_path)
        for epoch in range(args['num_epochs']):
            sess.run(tf.local_variables_initializer())
            for step in tqdm(range(step_up)):
                batch_base_loss, batch_loss, batch_eval, batch_y_, batch_y, _ = sess.run(
                    [train_base_loss, train_loss, train_eval_metric_ops, train_y_, train_y, train_op])
                print('#Epoch:{}, #batch:{}, loss:{:.4f} base_loss_1:{:.4f} acc_1:{:.4f} auc_1:{:.4f}'
                      ' base_loss_2:{:.4f} acc_2:{:.4f} auc_2:{:.4f}'
                      .format(epoch, step, batch_loss, batch_base_loss[0], batch_eval['acc1'][1], batch_eval['auc1'][1],
                              batch_base_loss[1], batch_eval['acc2'][1], batch_eval['auc2'][1]))
            print()
            print("-----------------Validating Start-----------------\n")
            sess.run(tf.local_variables_initializer())
            base_loss, loss = [0, 0], 0
            valid_step = 200
            for _ in tqdm(range(valid_step)):
                batch_base_loss, batch_loss, batch_eval, batch_y_, batch_y = sess.run(
                    [valid_base_loss, valid_loss, valid_eval_metric_ops, valid_y_, valid_y])
                base_loss = [batch_base_loss[i] + base_loss[i] for i in range(len(base_loss))]
                loss += batch_loss
            base_loss = [i / valid_step for i in base_loss]
            loss /= valid_step
            print('#Validated loss:{:.4f} base_loss_1:{:.4f} acc_1:{:.4f} auc_1:{:.4f}'
                  ' base_loss_2:{:.4f} acc_2:{:.4f} auc_2:{:.4f}'
                  .format(loss, base_loss[0], batch_eval['acc1'][1], batch_eval['auc1'][1],
                          base_loss[1], batch_eval['acc2'][1], batch_eval['auc2'][1]))
            print(batch_y_[:100])
            print(batch_y[:100])
            if batch_eval['auc1'][1] + batch_eval['auc2'][1] > best_val_auc:
                print("Get new best result!")
                best_val_auc = batch_eval['auc1'][1] + batch_eval['auc2'][1]
                saver.save(sess, model_path)
                print("New best result Saved!")
        print("-----------------Testing Start-----------------\n")
        sess.run(tf.local_variables_initializer())
        step = 0
        base_loss, loss = [0, 0], 0
        while True:
            try:
                step += 1
                batch_base_loss, batch_loss, batch_eval, batch_y_, batch_y = sess.run(
                    [test_base_loss, test_loss, test_eval_metric_ops, test_y_, test_y])
                base_loss = [batch_base_loss[i] + base_loss[i] for i in range(len(base_loss))]
                loss += batch_loss
                print('#Testing step:{} loss:{:.4f} base_loss_1:{:.4f} acc_1:{:.4f} auc_1:{:.4f}'
                      ' base_loss_2:{:.4f} acc_2:{:.4f} auc_2:{:.4f}'
                      .format(step, batch_loss, batch_base_loss[0], batch_eval['acc1'][1], batch_eval['auc1'][1],
                              batch_base_loss[1], batch_eval['acc2'][1], batch_eval['auc2'][1]), end='\r')
            except tf.errors.OutOfRangeError:
                break
            print()
        print('Average base_loss_1:{:.4f} base_loss_2:{:.4f} loss:{:.4f}'.format(base_loss[0] / step, base_loss[1] / step,
                                                                          loss / step))


if __name__ == "__main__":
    import argparse
    from configure import get_exp_configure

    parser = argparse.ArgumentParser(description="STARec")
    parser.add_argument(
        "-m", "--model", type=str, choices=['STARec', 'LSTM'], default='STARec', help="Model to use")
    parser.add_argument(
        '-d', '--dataset', type=str, choices=["tmall"], default="tmall", help="Dataset to use")
    parser.add_argument('--data_dir', type=str, default='../Data')
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--load_model', action="store_true", default=False)
    parser.add_argument('--use_label', action="store_true", default=False)
    parser.add_argument('--hard_search', action="store_true", default=False)
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate')
    parser.add_argument("-c", "--cuda", type=str, default="0")
    parser.add_argument("--postfix", type=str, default="", help="a string appended to the file name of the saved model")
    parser.add_argument("--rand_seed", type=int, default=-1, help="random seed for torch and numpy")
    args = parser.parse_args().__dict__
    args["exp_name"] = "_".join([args["model"], args["dataset"]])
    args.update(get_exp_configure(args))
    if args["cuda"] == "none":
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = args["cuda"]
    main(args)
