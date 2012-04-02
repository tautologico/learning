# prepare the Iris dataset for training and testing

def process_iris():
    dfile = open('iris.data')
    lines = dfile.readlines()
    dfile.close()
    lines_train = lines[0:35] + lines[50:85] + lines[100:135]
    lines_test = lines[35:50] + lines[85:100] + lines[135:150]
    train_file = open('iris.train', 'w')
    train_file.writelines(lines_train)
    train_file.close()
    test_file = open('iris.test', 'w')
    test_file.writelines(lines_test)
    test_file.close()


if __name__ == '__main__':
    process_iris()
