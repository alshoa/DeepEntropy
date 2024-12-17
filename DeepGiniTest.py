from sklearn.preprocessing import LabelBinarizer
from scipy.io import loadmat
from keras.datasets import mnist, cifar10, fashion_mnist, cifar100
from keras.models import load_model
from keras.utils import np_utils
import os
import argparse
import keras
from keras.callbacks import ModelCheckpoint
from termcolor import colored
from ATS.ATS import ATS
from utils import model_conf
import numpy as np
from utils.utils import num_to_str, shuffle_data, shuffle_data3
from entropy import calculate_entropy,probility_entropy_weight1,gini_score


def train_model(model, filepath, X_train, Y_train, X_test, Y_test, epochs=10, verbose=1):
    checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_accuracy', mode='auto',
                                 save_best_only='True')
    model.fit(X_train, Y_train, batch_size=128, epochs=epochs, validation_data=(X_test, Y_test),
              callbacks=[checkpoint],
              verbose=verbose)
    model = load_model(filepath)
    return model


def get_psedu_label(m, x):
    pred_test_prob = m.predict(x)
    y_test_psedu = np.argmax(pred_test_prob, axis=1)
    return y_test_psedu


def diverse_errors_num(y_s, y_psedu):
    fault_pair_arr = []
    fault_idx_arr = []
    for ix, (y_s_temp, y_psedu_temp) in enumerate(zip(y_s, y_psedu)):
        if y_s_temp == -1:
            continue
        elif y_s_temp == y_psedu_temp:
            continue
        else:
            key = (y_s_temp, y_psedu_temp)
            if key not in fault_pair_arr:
                fault_pair_arr.append(key)
                fault_idx_arr.append(ix)
    return len(fault_idx_arr)


def get_tests(x_dau, y_dau, order):
    x_sel, y_sel = x_dau[:test_size // 2], y_dau[:test_size // 2]
    order1 = order[:test_size // 2]
    x_val, y_val = x_dau[test_size // 2:], y_dau[test_size // 2:]
    return x_sel, y_sel, x_val, y_val, order1


def fault_detection(y, y_psedu):
    fault_num = np.sum(y != y_psedu)
    #print("发现错误总数: {}".format(fault_num))

    diverse_fault_num = diverse_errors_num(y, y_psedu)
    #print("发现错误种类数: {}/{}".format(diverse_fault_num, 90))
    return fault_num, diverse_fault_num


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", "-d", help="Dataset", type=str, default="mnist")
    parser.add_argument(
        "--m", "-m", help="Model_name", type=str, default="LeNet5"
    )
    parser.add_argument(
        "--ID",
        "-ID",
        help="The ID of running",
        type=int,
        default=0,
    )
    args = parser.parse_args()
    assert args.d in ["mnist", "Cifar10", "Fashion_mnist", "SVHN"], "Dataset should be either 'mnist' or 'cifar10'"
    assert args.m in ["LeNet1", "LeNet4", "LeNet5", "12Conv", "ResNet20"], "Model should be either 'LeNet1' or 'LeNet5'"

    print(args)

    # initial ATS
    base_path = "demo"
    os.makedirs(base_path, exist_ok=True)

    data_name = args.d
    Model_name = args.m
    IDD = args.ID

    CLIP_MIN = -0.5
    CLIP_MAX = 0.5

    # mnist data
    ##
    # color_print("load LeNet-5 model and MNIST data sets", "blue")
    # print(dau)
    # (x_train, _), (x_test, y_test) = dau.load_data(use_norm=True)
    ##MNISt
    if data_name == "mnist":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
        x_train = x_train.astype("float32")
        x_test = x_test.astype("float32")
        if Model_name == "LeNet5":
            index_without_noisy = np.load(
                r"C:\Users\alshoa\Desktop\DeepEntropy-Prioritizing-Robustness-Testing-of-Deep-Classifiers--main\ATS-master_final\Index_WN\mnist\Inw_mnist_LeNet5.npy")
            model_path = model_conf.get_model_path(model_conf.mnist, model_conf.LeNet5)
        if Model_name == "LeNet1":
            index_without_noisy = np.load(
                r"C:\Users\alshoa\Desktop\DeepEntropy-Prioritizing-Robustness-Testing-of-Deep-Classifiers--main\ATS-master_final\Index_WN\mnist\Inw_mnist_LeNet1.npy")
            model_path = model_conf.get_model_path(model_conf.mnist, model_conf.LeNet1)

    if data_name == "Cifar10":
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype("float32")
        x_test = x_test.astype("float32")
        if Model_name == "12Conv":
            # index_without_noisy=np.load("/content/drive/MyDrive/A_Paper2/A_Paper2/ATS-master_final/Index_WN/cifar10/Inw_cifar10_12Conv.npy")
            index_without_noisy = np.load(
                r"C:\Users\alshoa\Desktop\DeepEntropy-Prioritizing-Robustness-Testing-of-Deep-Classifiers--main\ATS-master_final\Index_WN\cifar10\Inw_cifar10_12Conv.npy")
            model_path = model_conf.get_model_path(model_conf.cifar10, model_conf.Conv12)
        if Model_name == "ResNet20":
            index_without_noisy = np.load(
                r"C:\Users\alshoa\Desktop\DeepEntropy-Prioritizing-Robustness-Testing-of-Deep-Classifiers--main\ATS-master_final\Index_WN\cifar10\Inw_cifar10_ResNet20.npy")
            model_path = model_conf.get_model_path(model_conf.cifar10, model_conf.ResNet20)
    if data_name == "SVHN":
        train_raw = loadmat(
            r'C:\Users\alshoa\Desktop\DeepEntropy-Prioritizing-Robustness-Testing-of-Deep-Classifiers--main\Data\train_32x32.mat')
        test_raw = loadmat(
            r'C:\Users\alshoa\Desktop\DeepEntropy-Prioritizing-Robustness-Testing-of-Deep-Classifiers--main\Data\test_32x32.mat')
        x_train = np.array(train_raw['X'])
        x_test = np.array(test_raw['X'])
        y_train = train_raw['y']
        y_test = test_raw['y']
        x_train = np.moveaxis(x_train, -1, 0)
        x_test = np.moveaxis(x_test, -1, 0)
        x_test = x_test.reshape(-1, 32, 32, 3)
        x_train = x_train.reshape(-1, 32, 32, 3)
        x_train = x_train.astype("float32")
        x_test = x_test.astype("float32")
        lb = LabelBinarizer()
        y_train = lb.fit_transform(y_train)
        y_test = lb.fit_transform(y_test)
        y_test = np.argmax(y_test, axis=1)
        if Model_name == "LeNet5":
            index_without_noisy = np.load(
                r"C:\Users\alshoa\Desktop\DeepEntropy-Prioritizing-Robustness-Testing-of-Deep-Classifiers--main\ATS-master_final\Index_WN\SVHN\Inw_SVHN_LeNet5.npy")
            model_path = model_conf.get_model_path(model_conf.svhn, model_conf.LeNet5)
    if data_name == "Fashion_mnist":
        # load dataset
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
        x_train = x_train.astype("float32")
        x_test = x_test.astype("float32")
        if Model_name == "LeNet4":
            index_without_noisy = np.load(
                r"C:\Users\alshoa\Desktop\DeepEntropy-Prioritizing-Robustness-Testing-of-Deep-Classifiers--main\ATS-master_final\Index_WN\Fashion_mnist\Inw_Fashion_mnist_LeNet4.npy")
            model_path = model_conf.get_model_path(model_conf.fashion, model_conf.LeNet4)

    print("load  model and data sets", "blue")
    ##
    if data_name != "SVHN":
        y_test = np_utils.to_categorical(y_test, 10)
        y_test = np.argmax(y_test, axis=1)
        y_train = np_utils.to_categorical(y_train, 10)
    x_train = (x_train / 255.0) - (1.0 - CLIP_MAX)
    x_test = (x_test / 255.0) - (1.0 - CLIP_MAX)

    test_size = len(x_test)
    print("测试集大小为", test_size)
    nb_classes = model_conf.fig_nb_classes

    ori_model = load_model(model_path)

    acc = ori_model.evaluate(x_test, keras.utils.np_utils.to_categorical(y_test, 10), verbose=0)[1]
    print("原始准确率为 {}".format(acc))
    # data augmentation
    print("data augmentation")
    x_dau, y_dau = x_test[index_without_noisy], y_test[index_without_noisy]
    print("模型数量和形状为", x_dau.shape)
    # ori_order=np.array(range(len(x_dau)))


    # 起始值为100，步长为100，结束值为1000
    for num in range(100, 1100, 100):

        total_mistake = 0
        mistake_type = 0
        for i in range(1, 21):
            x_dau, y_dau, t1_order = shuffle_data3(x_dau, y_dau, index_without_noisy)

            # selection
            x_sel, y_sel, x_val, y_val, t2_order = get_tests(x_dau, y_dau, t1_order)
            acc_val0 = ori_model.evaluate(x_val, keras.utils.np_utils.to_categorical(y_val, 10), verbose=0)[1]
            y_sel_psedu = get_psedu_label(ori_model, x_sel)
            # 使用模型预测测试用例的概率
            predicted_probabilities = ori_model.predict(x_sel)

            # 计算每个测试用例的熵
            entropies = np.array([gini_score(prob) for prob in predicted_probabilities])

            # 获取熵值和对应的索引
            entropy_indices = np.argsort(entropies)[::-1]  # 降序排序获取索引

            # 选择熵最高的前1000个测试用例的索引
            top_1000_indices = entropy_indices[:num]

            # 使用这些索引从原始测试用例中选择数据
            x_top_1000 = x_sel[top_1000_indices]
            y_top_1000 = y_sel[top_1000_indices]

            # 提取这1000个测试用例的熵值
            entropies_top_1000 = entropies[top_1000_indices]

            # 打印或处理选定的测试用例
            y_psedu_top_1000 = ori_model.predict(x_top_1000)

            # 将预测的概率转换为类别标签
            y_psedu_top_1000_labels = np.argmax(y_psedu_top_1000, axis=1)

            # 调用 fault_detection 函数
            fault_num, diverse_fault_num = fault_detection(y_top_1000, y_psedu_top_1000_labels)
            total_mistake = total_mistake + fault_num
            mistake_type = mistake_type + diverse_fault_num
            #print("本次结果为"+str(fault_num)+"和"+str(diverse_fault_num))
        print("检索前"+str(num)+"时平均结果为")
        print("平均发现错误总数"+str(total_mistake/20))
        print("平均发现错误种类"+str(mistake_type/20))