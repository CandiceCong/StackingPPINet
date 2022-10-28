#-*- encoding:utf8 -*-                                                                                                                   


class DefaultConfig(object):

    #通过热码编码的方式，将字母表示的氨基酸序列进行数字表示
    acid_one_hot = [0 for i in range(20)]
    acid_idex = {j:i for i,j in enumerate("ACDEFGHIKLMNPQRSTVWY")}
    
    #文件保存路径
    BASE_PATH = "../../"
    sequence_path = "{0}/data_cache/sequence_data".format(BASE_PATH)
    pssm_path = "{0}/data_cache/pssm_data".format(BASE_PATH)
    dssp_path = "{0}/data_cache/dssp_data".format(BASE_PATH)
    phyc_path = "{0}/data_cache/phyc_data".format(BASE_PATH)
    ach_path = "{0}/data_cache/ach_data".format(BASE_PATH)
    psea_path = "{0}/data_cache/psea_data".format(BASE_PATH)
    
    #序列的最大保留长度，去掉全局特征后，暂时没有用
    max_sequence_length = 500

    #局部特征的作用区间
    windows_size = [16,16,16]               
    
    batch_size = 32

    #原代码中训练集和验证集的划分比例，重新划分数据集后，暂时没用
    splite_rate = 0.9

    #特征维度
    seq_dim = 20
    dssp_dim = 9
    phyc_dim = 10
    ach_dim = 2
    pssm_dim = 20
    #伪氨基酸特征比较特殊，跟最大关联度相关，暂时设置为11，加上前20维，共30维
    psea_dim = 30

    #卷积模块
    cnn_chanel = [8,16,8]
    kernels = [7,5,3]

    #池化层
    pools = [3,3,3]
    stride = [2,1,2]

    #注意力机制模块，atten_dim必须是atten_head的倍数
    atten_dim = 16
    atten_head = 4
    
    dropout = 0.5
    #框架编号
    frame_num = 0
    #输出层结点数目，为1也就是进行两类，是或否的判断
    class_num = 1

    #定义两类特征的比重
    raio = 1


