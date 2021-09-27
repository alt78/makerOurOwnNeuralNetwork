#Код для создание трехслойной нейронки
#Данные с Mnist

import numpy
# сигмоида в scipy
import scipy.special

#check
#check
# Класс нейроной сети

class neuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # Задание количества узлов в входном, скрытом, выходном слоях
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # Матрицы вес коэф связей wih(между входным
        # и скрытыми слоями) и who(скрытым и выходным слоями)
        # исп нормальное распределение с центром в нуле, стандартное откл
        # вычисляет возведением количества скрытых слоев в стпеень -0.5
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        # кф обучение
        self.lr = learningrate

        # функция активации
        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    # тренировка нс
    def train(self, inputs_list, targets_list):
        # преобразование списка вх значений в 2м массив
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # расчитать вх сигналы для скрытого слоя
        hidden_inputs = numpy.dot(self.wih, inputs)
        # расчитать исх сигналы для скрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs)

        # расчитать вх сигналы для вых слоя
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # расчитать исх сигналы для вых слоя
        final_outputs = self.activation_function(final_inputs)

        # ошибка вых слоя = целевое - фактическое
        output_errors = targets - final_outputs
        # ошибки скрытого слоя это outp_e, которые распределяются процорц вес кф связей
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # обновляем веса между скрытым и выходым слоями
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                        numpy.transpose(hidden_outputs))


        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                        numpy.transpose(inputs))

        pass

    # опрос нс
    def query(self, inputs_list):

        inputs = numpy.array(inputs_list, ndmin=2).T

        # вычисляем вх сигн для скрытого слоя
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

# кол-во вх, скрытых, выходных узлов (тут оптимально 200 скр, но долго)
input_nodes = 784
hidden_nodes = 50
output_nodes = 10

learning_rate = 0.1

n = neuralNetwork(input_nodes,hidden_nodes,output_nodes, learning_rate)

#загрузка данных
training_data_file = open("mnist_dataset/mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()




# сколько раз тренируем
epochs = 1

for e in range(epochs):
    # перебор всех данных в наборе данных
    for record in training_data_list:
        all_values = record.split(',')
        # масштабируем и смещаем вх значения
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # создать целевые выходные значения (все равны 0.01, за искл маркерного 0.99
        targets = numpy.zeros(output_nodes) + 0.01
        # all_values[0] целевое маркерное значение
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
        pass
    pass


test_data_file = open("mnist_dataset/mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

# тестирование нс

# журнал оценок
scorecard = []

# перебрать все записи в текстовом наборе данных
for record in test_data_list:
    all_values = record.split(',')
    # правильный ответ-первое значение
    correct_label = int(all_values[0])
    # смещаем значение
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    # опрос
    outputs = n.query(inputs)
    # индекс наиб значения явл маркерным
    label = numpy.argmax(outputs)
    # добавляем оценку ответа к концу списка
    if (label == correct_label):

        scorecard.append(1)
    else:

        scorecard.append(0)
        pass

    pass


scorecard_array = numpy.asarray(scorecard)
print ("performance = ", scorecard_array.sum() / scorecard_array.size)