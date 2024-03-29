from ID3 import decision_tree
from utils import get_iris_data, get_wine_data


def main():
    features, train, test = get_iris_data(0.8)
    # features, train, test = get_wine_data(0.8)

    model = decision_tree(
        data=train,
        feature_names=features,
        max_depth=4,
        min_sample_split=10
    )
    model.train()
    acc, f1 = model.test(test[:, :-1], test[:, -1])
    print("accuracy : {} f1score : {}".format(acc, f1))
    model.print_decision_tree()


if __name__ == "__main__":
    main()
