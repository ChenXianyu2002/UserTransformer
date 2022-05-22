# STARec
使用*Pyotrch*对论文Learn from Past, Evolve for Future: Search-based Time-awareRecommendation with Sequential Behavior Data的模型以及其他Baselines模型的实现。

##数据集
使用了三个淘宝的公开数据集Tmall、Alipay和Taobao。

##各文件作用
*DealData.py*:对数据的预处理，包括：去除重复项，按用户id分组、同用户按时间排序、去除长度小于20的用户序列、为各个特征空间做重映射、记录每个category对应的users。

*数据格式*:

user_item.npz:包含了三个numpy数组——
    log:用户历史行为记录，数据处理后保证第一列是user，最后三列分别是category、time、label即可，中间列排列可随意。
    begin_len:用户历史行为长度信息，第一列是每个用户序列在log中开始位置，第二列是每个用户序列长度
    fields:从user到categories每个特征域下的特征数目，第一个数是users数目，最后一个数是categories数目。

category_users.npy:存储了每个category对应的用户，存储格式为以numpy.array为对象的numpy数组。若不使用其他用户作为辅助，则该文件可以不使用。


*STARec*目录下：
*BackModels.py*是各个RNN模型的实现。*configure.py*是参数配置文件。*utils.py*包含了evaluate函数与对预测模型的调用。
其余文件中，名字中带有Search的文件即为STARec实现版本，不带有的为其他Baseline实现版本；其中main*.py为主运行程序入口；*Dataloader.py即自实现的Dataloader；*PredictModel.py是预测模型，包含了RNN模型的调用、embedding处理、时间差计算、label_trick处理、相关度计算等。

