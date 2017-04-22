
from sklearn.metrics.pairwise import cosine_similarity

class Eval():
    def __init__(self, name):
        self.name = name

    def _get_cos_sim_mx(emb):
        cnt = emb.shape[0]
        mx = np.ndarray(shape=(cnt, cnt), dtype=np.float32)
        for i in range(0, cnt):
            for j in range(0, i + 1):
                sim = cosine_similarity(emb[i].reshape(1, -1), emb[j].reshape(1, -1))
                mx[i][j] = sim
                mx[j][i] = sim
        return mx

    def evalute(self, input):
        raise NotImplementedError

    def save_result(self, header, result, fn):
        raise NotImplementedError