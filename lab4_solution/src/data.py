from sklearn.datasets import make_classification, make_moons
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def split_scale(X, y, seed=42):
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y,
        train_size=0.60,
        random_state=seed,
        stratify=y,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp,
        test_size=0.50,
        random_state=seed,
        stratify=y_tmp,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    X_all = scaler.transform(X)

    result = {
        "X": X_all,
        "y": y,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "scaler": scaler,
    }

    if X_all.shape[1] == 2:
        result["X_plot"] = X_all[:, :2]
        result["plot_kind"] = "features"
        result["pca"] = None
    else:
        pca = PCA(n_components=2, random_state=seed)
        result["X_plot"] = pca.fit_transform(X_all)
        result["plot_kind"] = "pca"
        result["pca"] = pca

    return result


def make_lab_datasets(seed=467866):
    X_moons, y_moons = make_moons(
        n_samples=400,
        noise=0.15,
        random_state=seed,
    )

    X_class, y_class = make_classification(
        n_samples=200,
        n_features=5,
        n_redundant=2,
        n_informative=2,
        n_clusters_per_class=2,
        n_classes=2,
        random_state=seed,
    )

    datasets = {
        "moons": split_scale(X_moons, y_moons, seed),
        "classification": split_scale(X_class, y_class, seed),
    }
    datasets["moons"].update({"name": "make_moons", "original_shape": X_moons.shape})
    datasets["classification"].update({"name": "make_classification", "original_shape": X_class.shape})
    return datasets
