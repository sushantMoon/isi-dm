from ID3 import decision_tree
from utils import get_iris_data, get_wine_data


def main():
    features, train, test = get_iris_data(0.9)
    # features, train, test = get_wine_data(0.8)

    model = decision_tree(
        data=train,
        feature_names=features,
        max_depth=4,
        min_sample_split=10
    )
    model.train()
    print(model.test(test[:, :-1], test[:, -1]))


if __name__ == "__main__":
    main()
