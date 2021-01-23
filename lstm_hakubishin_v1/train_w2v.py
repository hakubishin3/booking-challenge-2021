import numpy as np
import multiprocessing
from gensim.models import Word2Vec
from sklearn import preprocessing
from src import log, set_out, span, load_train_test_set


def cos_sim(v1: np.ndarray, v2: np.ndarray) -> float:
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


if __name__ == '__main__':
    train_test = load_train_test_set({"input_dir_path": "./data/input/"})
    target_le = preprocessing.LabelEncoder()
    train_test['city_id'] = target_le.fit_transform(train_test['city_id'])
    train_test['city_id'] = train_test['city_id'].astype(str)

    train = train_test[train_test['row_num'].isnull()]
    train_trips = train[train['city_id'] != train['city_id'].shift(1)].groupby('utrip_id')['city_id'].apply(lambda x: x.values).reset_index()

    train_trips['n_trips'] = train_trips['city_id'].map(lambda x: len(x))

    cores = multiprocessing.cpu_count()
    w2v_model = Word2Vec(
        min_count=1,
        window=5,
        size=300,
        sample=6e-5,
        alpha=0.03,
        min_alpha=0.0007,
        negative=20,
        workers=cores - 1,
        compute_loss=True,
    )
    sentences = [list(ar) for ar in train_trips['city_id'].to_list()]
    w2v_model.build_vocab(sentences)
    for epoch in range(20):
        w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=w2v_model.iter)
        # print(w2v_model.get_latest_training_loss())
    w2v_model.save("./data/output/word2vec/word2vec.model")

    res = []
    for i in range(0, 39902):
        if i == 0:
            # city_id = 0
            vec = np.zeros(300)
        else:
            # city_id <> 0
            vec = w2v_model[str(i)]
        res.append(vec)
    res = np.stack(res)

    import pdb; pdb.set_trace()
