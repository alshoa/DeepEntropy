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
from entropy import calculate_entropy
import random

def initialize_population(pop_size):
    population = []
    for _ in range(pop_size):
        w1 = np.random.rand()
        w2 = 1 - w1
        population.append((w1, w2))
    return population


def fitness(w1, w2, predicted_probabilities, y_top_1000):
    """
    计算给定w1和w2的适应度值（APFD）。
    w1和w2分别是两个权重参数，predicted_probabilities是模型预测的概率，
    y_top_1000是前1000个测试用例的实际标签。
    """

    # 计算每个测试用例的熵
    entropies = np.array([DeepEntropy(w1, w2, prob) for prob in predicted_probabilities])

    # 对测试用例按照熵值进行降序排序
    entropy_indices = np.argsort(entropies)[::-1]  # 按照熵降序排列

    # 确保选择的测试用例数量不超过实际测试用例数目
    num = min(1000, len(entropy_indices))  # 取1000和实际数据集大小中的最小值
    top_1000_indices = entropy_indices[:num]  # 选择前num个测试用例的索引

    # 确保 top_1000_indices 不会超出 y_top_1000 的范围
    num_samples = len(y_top_1000)  # 获取 y_top_1000 的长度
    top_1000_indices = top_1000_indices[top_1000_indices < num_samples]  # 确保索引不超过 y_top_1000 的大小

    # 使用这些索引从原始测试用例中选择数据
    x_top_1000 = predicted_probabilities[top_1000_indices]
    y_top_1000_sorted = y_top_1000[top_1000_indices]  # 按照熵排序后的真实标签

    # 预测前1000个测试用例的标签
    y_psedu_top_1000 = np.argmax(x_top_1000, axis=1)  # 将概率转换为预测标签

    # 计算APFD适应度
    apfd_value = calculate_apfd(y_top_1000_sorted, y_psedu_top_1000,num)  # 计算APFD值

    return apfd_value


def selection(population, fitnesses):
    # 选择概率与适应度正相关
    total_fitness = sum(fitnesses)
    probabilities = [f / total_fitness for f in fitnesses]
    selected_idx = np.random.choice(len(population), size=2, replace=False, p=probabilities)
    return population[selected_idx[0]], population[selected_idx[1]]


def crossover(parent1, parent2):
    # 单点交叉
    if random.random() < 0.7:  # 设置交叉概率
        child1 = (parent1[0], parent2[1])
        child2 = (parent2[0], parent1[1])
    else:
        child1 = parent1
        child2 = parent2
    return child1, child2


def mutate(child, mutation_rate=0.01):
    # 变异
    if random.random() < mutation_rate:
        child1 = child[0] + np.random.uniform(-0.05, 0.05)
        child2 = 1 - child1
        return child1, child2
    return child


# 遗传算法主函数
def genetic_algorithm(predicted_probabilities, y_top_1000, pop_size=5, generations=50):
    # 初始化种群
    population = initialize_population(pop_size)

    # 开始演化
    for generation in range(generations):
        # 计算适应度
        fitnesses = [fitness(w1, w2, predicted_probabilities, y_top_1000) for w1, w2 in population]

        # 选择下一代
        new_population = []
        while len(new_population) < pop_size:
            parent1, parent2 = selection(population, fitnesses)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1)
            child2 = mutate(child2)
            new_population.append(child1)
            new_population.append(child2)

        population = new_population[:pop_size]

        # 计算当前种群最优适应度
        fitnesses = [fitness(w1, w2, predicted_probabilities, y_top_1000) for w1, w2 in population]
        best_idx = np.argmax(fitnesses)
        print(f"Generation {generation + 1}, Best APFD: {fitnesses[best_idx]}")

    # 返回最优权重
    fitnesses = [fitness(w1, w2, predicted_probabilities, y_top_1000) for w1, w2 in population]
    best_idx = np.argmax(fitnesses)
    return population[best_idx]

def DeepEntropy(a, b, Output_probability):
    Output_probability = np.ravel(Output_probability)  # 确保是一个一维数组
    max_prob = np.max(Output_probability)
    min_prob = np.min(Output_probability)
    entropy = calculate_entropy(Output_probability)
    result = max_prob - min_prob
    return a * entropy + b * (1 - result)


# 计算APFD
def calculate_apfd(sorted_test_cases, defect_detection_order):
    m = len(defect_detection_order)  # 缺陷数
    n = len(sorted_test_cases)  # 测试用例数
    apfd_sum = 0
    for i in range(m):
        # 查找与 defect_detection_order[i] 匹配的测试用例
        matched_indices = np.where(sorted_test_cases == defect_detection_order[i])[0]
        if matched_indices.size > 0:  # 如果找到了匹配的测试用例
            T_i = matched_indices[0] + 1  # 获取第一个匹配的索引（加1是为了符合APFD公式）
            apfd_sum += T_i
        else:
            print(f"No match found for defect {defect_detection_order[i]} in sorted_test_cases.")
            # 这里可以选择处理没有找到匹配项的情况，例如跳过该缺陷或者给定默认值
    apfd = 1 - (apfd_sum / (m * n)) +1/2*n # APFD公式
    return apfd





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

# def fault_detection(y, y_psedu):
#     #fault_num = np.sum(y != y_psedu)
#     #print("总错误数量: {}".format(fault_num))
#
#     diverse_fault_num = diverse_errors_num(y, y_psedu)
#     #print("揭示错误总数: {}/{}".format(diverse_fault_num, 90))
#     return fault_num, diverse_fault_num

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
                fault_idx_arr.append(ix)  # 记录新错误类型首次出现的位置
    return sum(fault_idx_arr), len(fault_idx_arr)  # 返回位置总和和新错误类型的数量


def calculate_apfd(y_true, y_pred, num):
    # 调用 diverse_errors_num 函数获取错误位置总和和不同错误类型的数量
    sum_of_indices, num_diverse_errors = diverse_errors_num(y_true, y_pred)

    # 计算 APFD
    n = num  # 测试用例数目
    m = num_diverse_errors  # 不同错误类型的数量
    apfd = 1 - (sum_of_indices / (n * m)) + (1 / (2 * n))

    return apfd

# def diverse_errors_num(y_s, y_psedu):
#     fault_pair_arr = []
#     fault_idx_arr = []
#     for ix, (y_s_temp, y_psedu_temp) in enumerate(zip(y_s, y_psedu)):
#         if y_s_temp == -1:
#             continue
#         elif y_s_temp == y_psedu_temp:
#             continue
#         else:
#             key = (y_s_temp, y_psedu_temp)
#             if key not in fault_pair_arr:
#                 fault_pair_arr.append(key)
#                 fault_idx_arr.append(ix)
#     return len(fault_idx_arr)


def get_tests(x_dau, y_dau, order):
    x_sel, y_sel = x_dau[:test_size // 2], y_dau[:test_size // 2]
    order1 = order[:test_size // 2]
    x_val, y_val = x_dau[test_size // 2:], y_dau[test_size // 2:]
    return x_sel, y_sel, x_val, y_val, order1




# def fault_detection(y, y_psedu):
#     #fault_num = np.sum(y != y_psedu)
#     #print("总错误数量: {}".format(fault_num))
#
#     diverse_fault_num = diverse_errors_num(y, y_psedu)
#     #print("揭示错误总数: {}/{}".format(diverse_fault_num, 90))
#     return fault_num, diverse_fault_num




# # 使用数值梯度的 gradient_descent 函数
# def gradient_descent(x_sel, y_sel, predicted_probabilities, num, learning_rate, iterations):
#     w1 = 0.5  # 初始值，因为 w1 + w2 = 1
#     for _ in range(iterations):
#         grad_w1 = numerical_gradient(w1, x_sel, y_sel, predicted_probabilities, num)
#         w1 -= learning_rate * grad_w1
#         # 确保 w1 在 0 和 1 之间
#         w1 = max(0, min(1, w1))
#     return w1



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

    # test_size = len(x_test)
    # nb_classes = model_conf.fig_nb_classes
    # ori_model = load_model(model_path)
    # acc = ori_model.evaluate(x_test, keras.utils.np_utils.to_categorical(y_test, 10), verbose=0)[1]
    # # data augmentation
    # x_dau, y_dau = x_test[index_without_noisy], y_test[index_without_noisy]
    # print("模型数量和形状为", x_dau.shape)
    # a = np.random.rand()  # 随机初始化 a 的值
    # learning_rate = 0.06  # 学习率
    # num = 1000  # 选择的前1000个测试用例
    # iterations = 100  # 训练的次数
    #
    # for time in range(50):  # 外部的迭代
    #     total_mistake = 0
    #     mistake_type = 0
    #     # 计算每个测试用例的熵
    #     x_dau, y_dau, t1_order = shuffle_data3(x_dau, y_dau, index_without_noisy)
    #
    #     # selection
    #     x_sel, y_sel, x_val, y_val, t2_order = get_tests(x_dau, y_dau, t1_order)
    #     acc_val0 = ori_model.evaluate(x_val, keras.utils.np_utils.to_categorical(y_val, 10), verbose=0)[1]
    #     y_sel_psedu = get_psedu_label(ori_model, x_sel)
    #
    #     # 使用模型预测测试用例的概率
    #     predicted_probabilities = ori_model.predict(x_sel)
    #
    #     # 计算熵
    #     entropies = np.array([DeepEntropy(a, 1 - a, prob) for prob in predicted_probabilities])
    #
    #     # 获取熵值和对应的索引
    #     entropy_indices = np.argsort(entropies)[::-1]  # 降序排序获取索引
    #     # 选择熵最高的前1000个测试用例的索引
    #     top_1000_indices = entropy_indices[:num]
    #     # 使用这些索引从原始测试用例中选择数据
    #     x_top_1000 = x_sel[top_1000_indices]
    #     y_top_1000 = y_sel[top_1000_indices]
    #     entropies_top_1000 = entropies[top_1000_indices]
    #
    #     # 打印或处理选定的测试用例
    #     y_psedu_top_1000 = ori_model.predict(x_top_1000)
    #
    #     # 将预测的概率转换为类别标签
    #     y_psedu_top_1000_labels = np.argmax(y_psedu_top_1000, axis=1)
    #
    #     # 计算当前的 APFD 值
    #     apfd_value = calculate_apfd(y_top_1000, y_psedu_top_1000_labels,num)
    #
    #     # 打印当前的 APFD 值
    #     print(f"Iteration {time + 1}, APFD: {apfd_value}, a: {a}")
    #
    #     # 梯度计算：数值梯度估计
    #     epsilon = 1e-5  # 很小的扰动，用于计算梯度
    #     a_plus = a + epsilon  # a 的一个小扰动
    #     a_minus = a - epsilon  # 另一个小扰动
    #
    #     # 计算扰动后的 APFD
    #     entropies_plus = np.array([DeepEntropy(a_plus, 1 - a_plus, prob) for prob in predicted_probabilities])
    #     entropy_indices_plus = np.argsort(entropies_plus)[::-1]
    #     top_1000_indices_plus = entropy_indices_plus[:num]
    #     x_top_1000_plus = x_sel[top_1000_indices_plus]
    #     y_top_1000_plus = y_sel[top_1000_indices_plus]
    #     entropies_top_1000_plus = entropies_plus[top_1000_indices_plus]
    #     y_psedu_top_1000_plus = ori_model.predict(x_top_1000_plus)
    #     y_psedu_top_1000_labels_plus = np.argmax(y_psedu_top_1000_plus, axis=1)
    #     apfd_value_plus = calculate_apfd(y_top_1000_plus, y_psedu_top_1000_labels_plus,num)
    #
    #     entropies_minus = np.array([DeepEntropy(a_minus, 1 - a_minus, prob) for prob in predicted_probabilities])
    #     entropy_indices_minus = np.argsort(entropies_minus)[::-1]
    #     top_1000_indices_minus = entropy_indices_minus[:num]
    #     x_top_1000_minus = x_sel[top_1000_indices_minus]
    #     y_top_1000_minus = y_sel[top_1000_indices_minus]
    #     entropies_top_1000_minus = entropies_minus[top_1000_indices_minus]
    #     y_psedu_top_1000_minus = ori_model.predict(x_top_1000_minus)
    #     y_psedu_top_1000_labels_minus = np.argmax(y_psedu_top_1000_minus, axis=1)
    #     apfd_value_minus = calculate_apfd(y_top_1000_minus, y_psedu_top_1000_labels_minus,num)
    #
    #     # 数值梯度估计
    #     gradient = (apfd_value_plus - apfd_value_minus) / (2 * epsilon)
    #
    #     # 更新 a
    #     a = a + learning_rate * gradient
    #
    #     # 确保 a 在 [0, 1] 范围内
    #     a = np.clip(a, 0, 1)
    #
    #     # 打印每次更新后的 a 和 APFD 值
    #     print(f"Updated a: {a}, APFD: {apfd_value}")
    test_size = len(x_test)
    nb_classes = model_conf.fig_nb_classes
    ori_model = load_model(model_path)
    acc = ori_model.evaluate(x_test, keras.utils.np_utils.to_categorical(y_test, 10), verbose=0)[1]

    # 初始化参数
    x_dau, y_dau = x_test[index_without_noisy], y_test[index_without_noisy]
    print("模型数量和形状为", x_dau.shape)
    a1 = np.random.rand()
    a2 = np.random.rand()
    learning_rate = 0.1  # 学习率
    num = 1000  # 选择的前1000个测试用例
    iterations = 2500  # 训练的次数
    x_dau, y_dau, t1_order = shuffle_data3(x_dau, y_dau, index_without_noisy)

    # selection
    x_sel, y_sel, x_val, y_val, t2_order = get_tests(x_dau, y_dau, t1_order)
    acc_val0 = ori_model.evaluate(x_val, keras.utils.np_utils.to_categorical(y_val, 10), verbose=0)[1]
    y_sel_psedu = get_psedu_label(ori_model, x_sel)

    # 使用模型预测测试用例的概率
    predicted_probabilities = ori_model.predict(x_sel)


    for time in range(iterations):  # 外部的迭代
        total_mistake = 0
        mistake_type = 0

        # 计算每个测试用例的熵


        # 计算熵
        entropies = np.array([DeepEntropy(a1, a2, prob) for prob in predicted_probabilities])

        # 获取熵值和对应的索引
        entropy_indices = np.argsort(entropies)[::-1]  # 降序排序获取索引
        # 选择熵最高的前1000个测试用例的索引
        top_1000_indices = entropy_indices[:num]
        x_top_1000 = x_sel[top_1000_indices]
        y_top_1000 = y_sel[top_1000_indices]

        # 打印或处理选定的测试用例
        y_psedu_top_1000 = ori_model.predict(x_top_1000)
        y_psedu_top_1000_labels = np.argmax(y_psedu_top_1000, axis=1)

        # 计算当前的 APFD 值
        apfd_value = calculate_apfd(y_top_1000, y_psedu_top_1000_labels, num)

        # 打印当前的 APFD 值
        print(f"Iteration {time + 1}, APFD: {apfd_value}, a1: {a1}, a2: {a2}")

        # 梯度计算：数值梯度估计
        epsilon = 1e-1
        a1_plus, a1_minus = a1 + epsilon, a1 - epsilon
        a2_plus, a2_minus = a2 + epsilon, a2 - epsilon

        # 扰动后重新计算 APFD
        entropies_plus = np.array([DeepEntropy(a1_plus, a2, prob) for prob in predicted_probabilities])
        apfd_value_plus = calculate_apfd(
            y_sel[np.argsort(entropies_plus)[::-1][:num]],
            np.argmax(ori_model.predict(x_sel[np.argsort(entropies_plus)[::-1][:num]]), axis=1),
            num,
        )

        entropies_minus = np.array([DeepEntropy(a1_minus, a2, prob) for prob in predicted_probabilities])
        apfd_value_minus = calculate_apfd(
            y_sel[np.argsort(entropies_minus)[::-1][:num]],
            np.argmax(ori_model.predict(x_sel[np.argsort(entropies_minus)[::-1][:num]]), axis=1),
            num,
        )

        # 数值梯度估计
        gradient_a1 = (apfd_value_plus - apfd_value_minus) / (2 * epsilon)

        # 更新 a1 和 a2
        a1 = a1 + learning_rate * gradient_a1
        a1 = np.clip(a1, 0, 1)

        entropies_plus_a2 = np.array([DeepEntropy(a1, a2_plus, prob) for prob in predicted_probabilities])
        apfd_value_plus_a2 = calculate_apfd(
            y_sel[np.argsort(entropies_plus_a2)[::-1][:num]],
            np.argmax(ori_model.predict(x_sel[np.argsort(entropies_plus_a2)[::-1][:num]]), axis=1),
            num,
        )

        entropies_minus_a2 = np.array([DeepEntropy(a1, a2_minus, prob) for prob in predicted_probabilities])
        apfd_value_minus_a2 = calculate_apfd(
            y_sel[np.argsort(entropies_minus_a2)[::-1][:num]],
            np.argmax(ori_model.predict(x_sel[np.argsort(entropies_minus_a2)[::-1][:num]]), axis=1),
            num,
        )

        # 数值梯度估计 for a2
        gradient_a2 = (apfd_value_plus_a2 - apfd_value_minus_a2) / (2 * epsilon)

        # 更新 a2
        a2 = a2 + learning_rate * gradient_a2
        a2 = np.clip(a2, 0, 1)

    # 在循环结束后归一化 a1 和 a2
    total = a1 + a2
    a1 /= total
    a2 /= total

    # 输出最终归一化结果
    print(f"Final normalized values: a1 = {a1}, a2 = {a2}")