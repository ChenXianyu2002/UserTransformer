import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd


class DealData:
    def __init__(self, dataset='taobao', bound_len=5):
        super(DealData, self).__init__()
        self.dataset = dataset
        data_paths = {
            'taobao': './Data/TaoBao/taobao/raw_data',
            'tmall': './Data/TaoBao/tmall/raw_data',
            'alipay': './Data/TaoBao/alipay/raw_data',
            'amazon': './Data/Amazon/',
            'movie': './Data/ml-25m/'
        }
        self.data_path = data_paths[dataset]
        self.bound_len = bound_len
        self.deal_func_dict = {
            'taobao': self.deal_taobao,
            'tmall': self.deal_tmall,
            'alipay': self.deal_alipay,
            'amazon': self.deal_amazon,
            'movie': self.deal_movie
        }

    def deal_func(self):
        self.deal_func_dict[self.dataset]()

    def get_cate_user(self):
        file = np.load(os.path.join(self.data_path, 'user_item.npz'), allow_pickle=True)
        user_category = file['log'][:, [0, -3]]
        field_nums = file['fields']
        print(field_nums)
        category_user_list = [[] for i in range(field_nums[-1])]
        other_fields_num = np.sum(field_nums[:-1])
        begin_len = file['begin_len']
        print(begin_len)
        for i in range(field_nums[0]):
            categories = user_category[begin_len[i, 0]:begin_len[i, 0] + begin_len[i, 1], -1]
            if self.dataset == 'movie' or self.dataset == 'amazon':
                categories = np.concatenate(categories)
            categories = np.unique(categories) - other_fields_num
            for category in categories:
                category_user_list[category].append(i)
        np.save(os.path.join(self.data_path, 'category_users.npy'), np.array(category_user_list))

    def deal_taobao(self):
        csv_file = pd.read_csv(os.path.join(self.data_path, 'UserBehavior.csv'), header=None)
        # 分别按照用户，时间排序
        csv_file = csv_file.sort_values(by=[0, 4])
        # 交换列，将标签放到最后,时间特征倒数第二列，类别特征倒数第三列
        csv_file = csv_file[[0, 1, 2, 4, 3]]
        csv_file.rename(columns={4: 3, 3: 4}, inplace=True)
        csv_file[4].replace(['pv', 'cart', 'fav', 'buy'], [0, 1, 1, 1], inplace=True)

        np_file = csv_file.to_numpy()
        print("CSV end!")
        # 对不同用户切分
        is_begin = np.append([1], np.diff(np_file[:, 0])) != 0
        # np_file[:, -2] = np.where(is_begin, 0, np.append([0], np.diff(np_file[:, -2])))
        belong_inter = np.cumsum(is_begin) - 1  # 用户属于第几个区间
        begin_loc = np.where(is_begin)[0]
        behavior_len = np.diff(np.append(begin_loc, [np_file.shape[0]]))
        print(np_file.shape)
        # 过滤掉长度较短的序列
        np_file = np_file[behavior_len[belong_inter] >= self.bound_len]
        print(np_file.shape)
        behavior_len = behavior_len[behavior_len >= self.bound_len]
        begin_loc = np.append([0], np.cumsum(behavior_len)[:-1])

        field_num = np.shape(np_file)[1] - 2
        fields_feature_num = np.array([0] * field_num)  # 记录每个特征域最大特征数目
        # 对域做重映射
        for field in range(field_num):
            remap_index = np.unique(np_file[:, field], return_inverse=True)[1]
            fields_feature_num[field] = np.max(remap_index) + 1
            np_file[:, field] = remap_index + np.sum(fields_feature_num[:field])

        print(np_file)
        print(fields_feature_num)
        np.savez(os.path.join(self.data_path, 'user_item.npz'),
                 log=np_file,
                 begin_len=np.stack([begin_loc, behavior_len], axis=-1),
                 fields=fields_feature_num
                 )

    def deal_alipay(self):
        csv_file = pd.read_csv(os.path.join(self.data_path, 'ijcai2016_taobao.csv'))
        # 分别按照用户，时间排序
        csv_file = csv_file.sort_values(by=['use_ID', 'time'])
        # 交换列，将标签放到最后,时间特征倒数第二列，类别特征倒数第三列
        csv_file = csv_file[['use_ID', 'sel_ID', 'ite_ID', 'cat_ID', 'time', 'act_ID']]
        np_file = csv_file.to_numpy()
        print("CSV end!")
        np_file = np_file[np.sort(np.unique(np_file, axis=0, return_index=True)[1])]
        # 将时间替换成纯天数
        month_day = np_file[:, -2] % 10000
        month = month_day // 100
        day = month_day % 100
        days = np.append([0], np.cumsum([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30]))
        np_file[:, -2] = days[month - 1] + day
        # 对不同用户切分
        is_begin = np.append([1], np.diff(np_file[:, 0])) != 0
        # np_file[:, -2] = np.where(is_begin, 0, np.append([0], np.diff(np_file[:, -2])))
        belong_inter = np.cumsum(is_begin) - 1  # 用户属于第几个区间
        begin_loc = np.where(is_begin)[0]
        behavior_len = np.diff(np.append(begin_loc, [np_file.shape[0]]))
        print(np_file.shape)
        # 过滤掉长度较短的序列
        np_file = np_file[behavior_len[belong_inter] >= self.bound_len]
        print(np_file.shape)
        behavior_len = behavior_len[behavior_len >= self.bound_len]
        begin_loc = np.append([0], np.cumsum(behavior_len)[:-1])
        field_num = np.shape(np_file)[1] - 2
        fields_feature_num = np.array([0] * field_num)  # 记录每个特征域最大特征数目
        # 对域做重映射
        for field in range(field_num):
            remap_index = np.unique(np_file[:, field], return_inverse=True)[1]
            fields_feature_num[field] = np.max(remap_index) + 1
            np_file[:, field] = remap_index + np.sum(fields_feature_num[:field])
            if field == field_num - 1:
                cate_count = np.bincount(remap_index)
                print(cate_count.min(), cate_count.max())
        print(np_file)
        print(fields_feature_num)
        np.savez(os.path.join(self.data_path, 'user_item.npz'),
                 log=np_file,
                 begin_len=np.stack([begin_loc, behavior_len], axis=-1),
                 fields=fields_feature_num
                 )

    def deal_tmall(self):
        user_log_csv = pd.read_csv(os.path.join(self.data_path, 'user_log_format1.csv'))
        user_info_csv = pd.read_csv(os.path.join(self.data_path, 'user_info_format1.csv'))
        # 分别按照用户，时间排序
        user_log_csv = user_log_csv.sort_values(by=['user_id', 'time_stamp'])
        # 用户信息按照用户id排序
        user_info_csv = user_info_csv.sort_values(by='user_id')
        # 交换列，将标签放到最后,时间特征倒数第二列，类别特征倒数第三列
        user_log_csv = user_log_csv[
            ['user_id', 'item_id', 'seller_id', 'brand_id', 'cat_id', 'time_stamp', 'action_type']]
        user_info_np = user_info_csv.to_numpy().astype(np.int32)
        user_log_np = user_log_csv.to_numpy().astype(np.int32)
        print("CSV end!")
        user_log_np = user_log_np[np.sort(np.unique(user_log_np, axis=0, return_index=True)[1])]
        user_log_np[:, -1][user_log_np[:, -1] == 3] = 1  # 0是点击，1是收藏添加，2是购买
        print(np.sum(user_log_np[:, -1]))
        # 将时间替换成纯天数
        month = user_log_np[:, -2] // 100
        day = user_log_np[:, -2] % 100
        print(month)
        days = np.append([0], np.cumsum([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30]))
        user_log_np[:, -2] = days[month - 1] + day
        # 对不同用户切分
        is_begin = np.append([1], np.diff(user_log_np[:, 0])) != 0
        # user_log_np[:, -2] = np.where(is_begin, 0, np.append([0], np.diff(user_log_np[:, -2])))
        belong_inter = np.cumsum(is_begin) - 1  # 用户属于第几个区间
        begin_loc = np.where(is_begin)[0]
        behavior_len = np.diff(np.append(begin_loc, [user_log_np.shape[0]]))
        print(user_log_np.shape)
        # 过滤掉长度较短的序列
        user_log_np = user_log_np[behavior_len[belong_inter] >= self.bound_len]
        print(user_log_np.shape)
        user_info_np = user_info_np[:len(behavior_len)][behavior_len >= self.bound_len]
        behavior_len = behavior_len[behavior_len >= self.bound_len]
        begin_loc = np.append([0], np.cumsum(behavior_len)[:-1])
        field_num = np.shape(user_log_np)[1] - 2
        fields_feature_num = np.array([0] * field_num)  # 记录每个特征域最大特征数目
        # 对域做重映射
        for field in range(field_num):
            remap_index = np.unique(user_log_np[:, field], return_inverse=True)[1]
            fields_feature_num[field] = np.max(remap_index) + 1
            user_log_np[:, field] = remap_index + np.sum(fields_feature_num[:field])
            if field == field_num - 1:
                user_log_np[:, field] = remap_index
        print(fields_feature_num)
        user_info_np = user_info_np[:, 1:]
        for field in range(2):
            remap_index = np.unique(user_info_np[:, field], return_inverse=True)[1]
            fields_feature_num = np.append(fields_feature_num, np.max(remap_index) + 1)
            user_info_np[:, field] = remap_index

        user_log_np = np.concatenate(
            [user_log_np[:, :-3], user_info_np[user_log_np[:, 0]], user_log_np[:, -3:]],
            axis=-1)

        fields_feature_num[[-3, -2, -1]] = fields_feature_num[[-2, -1, -3]]
        print(fields_feature_num)
        for field in [-3, -2, -1]:
            user_log_np[:, field - 2] += np.sum(fields_feature_num[:field])
        np.savez(os.path.join(self.data_path, 'user_item_multi.npz'),
                 log=user_log_np,
                 begin_len=np.stack([begin_loc, behavior_len], axis=-1),
                 fields=fields_feature_num
                 )

    def deal_amazon_to_csv(self):
        user_log_json = os.path.join(self.data_path, 'Electronics.json')
        item_info_json = os.path.join(self.data_path, 'meta_Electronics.json')
        useful_log_field = ['reviewerID', 'asin', 'style', 'vote', 'unixReviewTime', 'overall']
        useful_item_field = ['asin', 'price', 'main_cat', 'brand', 'category']
        log_csv = defaultdict(lambda: {})
        item_csv = defaultdict(lambda: {})
        i = 0
        with open(user_log_json) as user_log:
            for user_log_line in user_log:
                user_log_line = json.loads(user_log_line)
                for field in user_log_line.keys():
                    if field in useful_log_field:
                        log_csv[i][field] = user_log_line[field]
                i += 1
        log_csv = pd.DataFrame.from_dict(log_csv, orient='index')[useful_log_field]
        print(log_csv)
        log_csv.to_csv(os.path.join(self.data_path, 'user_log.csv'), index=False)
        del log_csv

        i = 0
        with open(item_info_json) as item_info:
            for item_info_line in item_info:
                item_info_line = json.loads(item_info_line)
                for field in item_info_line.keys():
                    if field in useful_item_field:
                        if field == 'category':
                            item_info_line[field].remove('Electronics')
                        item_csv[i][field] = item_info_line[field]
                i += 1

        item_csv = pd.DataFrame.from_dict(item_csv, orient='index')[useful_item_field]
        print(item_csv)
        item_csv.to_csv(os.path.join(self.data_path, 'item_info.csv'), index=False)

    def deal_amazon(self):
        user_log_csv = pd.read_csv(os.path.join(self.data_path, 'user_log.csv'))
        user_log_csv = user_log_csv[['reviewerID', 'asin', 'vote', 'unixReviewTime', 'overall']]
        user_log_csv['vote'].replace(np.nan, 0, inplace=True)
        user_log_csv = user_log_csv.astype(str)
        user_log_csv.sort_values(by=['reviewerID', 'unixReviewTime'], inplace=True)

        item_info_csv = pd.read_csv(os.path.join(self.data_path, 'item_info.csv'))
        # main_category = item_info_csv['main_cat'].to_numpy().astype(str)
        item_info_csv = item_info_csv[['asin', 'price', 'brand', 'main_cat', 'category']]
        item_info_csv['price'].replace(np.nan, '$0', inplace=True)

        print(user_log_csv.columns)
        print(item_info_csv.columns)
        user_log_np = user_log_csv.to_numpy()
        user_log_num = user_log_np.shape[0]
        item_info_np = item_info_csv.to_numpy()
        item_num = item_info_np.shape[0]
        print(user_log_num, item_num)
        item_info_np = item_info_np[np.unique(item_info_np[:, 0].astype(str), return_index=True)[1]]
        # 取商品id交集
        user_log_np = user_log_np[np.isin(user_log_np[:, 1].astype(str), item_info_np[:, 0].astype(str))]
        item_info_np = item_info_np[np.isin(item_info_np[:, 0].astype(str), user_log_np[:, 1].astype(str))]
        # 商品表按商品id排序
        item_info_np = item_info_np[np.argsort(item_info_np[:, 0].astype(str)), 1:]
        # 商品id重映射
        user_log_np[:, 1] = np.unique(user_log_np[:, 1].astype(str), return_inverse=True)[1]
        # 用户id重映射
        user_log_np[:, 0] = np.unique(user_log_np[:, 0].astype(str), return_inverse=True)[1]
        user_log_np[:, 2] = np.char.replace(user_log_np[:, 2].astype(str), ',', '')
        user_log_np[:, -1] = np.where(user_log_np[:, -1].astype(float) >= 5, 1, 0)
        user_log_np = user_log_np.astype(float).astype(int)
        # 对不同用户切分
        is_begin = np.append([1], np.diff(user_log_np[:, 0])) != 0
        # user_log_np[:, -2] = np.where(is_begin, 0, np.append([0], np.diff(user_log_np[:, -2])))
        belong_inter = np.cumsum(is_begin) - 1  # 用户属于第几个区间
        begin_loc = np.where(is_begin)[0]
        behavior_len = np.diff(np.append(begin_loc, [user_log_np.shape[0]]))
        print(user_log_np.shape)
        # 过滤掉长度较短的序列
        user_log_np = user_log_np[behavior_len[belong_inter] >= self.bound_len]
        print(user_log_np.shape)
        behavior_len = behavior_len[behavior_len >= self.bound_len]
        print(len(behavior_len))
        begin_loc = np.append([0], np.cumsum(behavior_len)[:-1])
        id_exist, item_id_remap = np.unique(user_log_np[:, 1], return_inverse=True)
        user_log_np[:, 1] = item_id_remap
        item_info_np = item_info_np[id_exist]

        def eval_and_len(i):
            list_i = eval(item_info_np[i, -1])
            list_i.append(item_info_np[i, -2])
            return [list_i, len(list_i)]

        # 字符串提取列表，并做重映射
        eval_len = np.array([eval_and_len(i) for i in range(item_info_np.shape[0])], dtype=object)
        item_info_np[:, -1] = eval_len[:, 0]
        eval_len = eval_len[:, 1]
        eval_squeeze = np.concatenate(item_info_np[:, -1])
        eval_squeeze = np.unique(eval_squeeze, return_inverse=True)[1]

        # print(eval_squeeze)
        # print(max(eval_len), min(eval_len), np.max(eval_squeeze))
        # 转换价格到数字
        item_info_np = item_info_np[:, 0:2]
        item_info_np[:, 0] = np.char.replace(item_info_np[:, 0].astype(str), '$', '')
        item_info_np[:, 0] = np.char.replace(item_info_np[:, 0].astype(str), ',', '')
        item_info_np[:, 0][
            np.logical_not(np.char.isnumeric(np.char.replace(item_info_np[:, 0].astype(str), '.', '')))] = 0
        item_info_np[:, 1] = np.unique(item_info_np[:, 1].astype(str), return_inverse=True)[1]
        item_info_np = item_info_np.astype(float)

        continuous_feature = np.stack([item_info_np[item_id_remap, 0], user_log_np[:, 2]], axis=-1)
        print(continuous_feature)
        user_log_np[:, 2] = item_info_np[item_id_remap, 1]
        user_log_np[:, 0] = np.unique(user_log_np[:, 0], return_inverse=True)[1]
        # 对域做重映射
        field_num = np.shape(user_log_np)[1] - 2 + 1
        fields_feature_num = np.array([0] * field_num)  # 记录每个特征域最大特征数目
        for field in range(field_num - 1):
            fields_feature_num[field] = np.max(user_log_np[:, field]) + 1
            user_log_np[:, field] = user_log_np[:, field] + np.sum(fields_feature_num[:field])
        fields_feature_num[-1] = np.max(eval_squeeze) + 1
        print(fields_feature_num)
        eval_squeeze = np.array(np.split(eval_squeeze + np.sum(fields_feature_num[:-1]), np.cumsum(eval_len)[:-1]),
                                dtype=np.ndarray)
        user_log_np = np.concatenate([user_log_np, np.expand_dims(eval_squeeze[item_id_remap], axis=-1)], axis=-1)
        user_log_np = user_log_np[:, [0, 1, 2, 5, 3, 4]]
        print(user_log_np)
        np.savez(os.path.join(self.data_path, 'user_item.npz'),
                 log=user_log_np,
                 begin_len=np.stack([begin_loc, behavior_len], axis=-1),
                 fields=fields_feature_num,
                 continuous_feature=continuous_feature
                 )

    def deal_movie(self):
        user_log_csv = pd.read_csv(os.path.join(self.data_path, 'ratings.csv'))
        item_info_csv = pd.read_csv(os.path.join(self.data_path, 'movies.csv'))
        print(user_log_csv.columns)
        print(item_info_csv.columns)

        user_log_csv = user_log_csv[['userId', 'movieId', 'timestamp', 'rating']]
        user_log_csv.sort_values(by=['userId', 'timestamp'], inplace=True)

        user_log_np = user_log_csv.to_numpy()
        item_info_np = item_info_csv['genres'].to_numpy().astype(str)
        user_log_np[:, -1] = np.where(user_log_np[:, -1] >= 4.5, 1, 0)
        user_log_np = user_log_np.astype(int)
        print(item_info_np)
        item_info_np = np.char.split(item_info_np, '|')
        print(item_info_np)
        split_len = [len(i) for i in item_info_np]
        item_info_np = np.unique(np.concatenate(item_info_np), return_inverse=True)[1]

        # 对不同用户切分
        is_begin = np.append([1], np.diff(user_log_np[:, 0])) != 0
        # user_log_np[:, -2] = np.where(is_begin, 0, np.append([0], np.diff(user_log_np[:, -2])))
        belong_inter = np.cumsum(is_begin) - 1  # 用户属于第几个区间
        begin_loc = np.where(is_begin)[0]
        behavior_len = np.diff(np.append(begin_loc, [user_log_np.shape[0]]))
        print(user_log_np.shape)
        # 过滤掉长度较短的序列
        user_log_np = user_log_np[behavior_len[belong_inter] >= self.bound_len]
        print(user_log_np.shape)
        behavior_len = behavior_len[behavior_len >= self.bound_len]
        print(len(behavior_len))
        begin_loc = np.append([0], np.cumsum(behavior_len)[:-1])

        user_log_np[:, 0] = np.unique(user_log_np[:, 0], return_inverse=True)[1]
        user_log_np[:, 1] = np.unique(user_log_np[:, 1], return_inverse=True)[1]

        # 对域做重映射
        field_num = np.shape(user_log_np)[1] - 2 + 1
        fields_feature_num = np.array([0] * field_num)  # 记录每个特征域最大特征数目
        for field in range(field_num - 1):
            fields_feature_num[field] = np.max(user_log_np[:, field]) + 1
            user_log_np[:, field] = user_log_np[:, field] + np.sum(fields_feature_num[:field])
        fields_feature_num[-1] = np.max(item_info_np) + 1
        print(fields_feature_num)
        item_info_np = np.array(np.split(item_info_np + np.sum(fields_feature_num[:-1]), np.cumsum(split_len)[:-1]),
                                dtype=np.ndarray)
        user_log_np = np.concatenate(
            [user_log_np, np.expand_dims(item_info_np[user_log_np[:, 1] - fields_feature_num[0]], axis=-1)], axis=-1)
        user_log_np = user_log_np[:, [0, 1, 4, 2, 3]]
        print(user_log_np)
        np.savez(os.path.join(self.data_path, 'user_item.npz'),
                 log=user_log_np,
                 begin_len=np.stack([begin_loc, behavior_len], axis=-1),
                 fields=fields_feature_num,
                 )


if __name__ == '__main__':
    d = DealData('tmall', bound_len=20)
    d.deal_func()
