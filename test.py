from train import clean_data
import pickle
import pandas as pd
import numpy as np
import sys, getopt


def main(argv):
    test_file = ''
    model = ''
    try:
        opts, args = getopt.getopt(argv, "ht:m:", ["test=", "model="])
    except getopt.GetoptError:
        print('test.py -t <test_file> -m <outputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -t <test_file> -m <outputfile>')
            sys.exit()
        elif opt in ("-t", "--test"):
            test_file = arg
        elif opt in ("-m", "--model"):
            model = arg
    print('Test file is ', test_file)
    print('Model file is ', model)


    df = pd.read_csv(test_file)
    X, _ = clean_data(df, mod="test")
    model = pickle.load(open(model, "rb"))
    predictions = model.predict(X)
    np.savetxt(model.__class__.__name__ + ".csv",
               predictions.astype(int),  fmt='%i')


if __name__ == "__main__":
    main(sys.argv[1:])
