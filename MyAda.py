import typing as tp
import sklearn
import numpy as np
import inspect


class MyAdaBoostClassifier: 
    """
    Multiclass AdaBoost implementation with SAMME.R algorithm 
    """
    big_number = 1 << 32 
    eps = 1e-8 
 
    def __init__( 
            self, 
            n_estimators: int, 
            base_estimator: tp.Type[sklearn.base.BaseEstimator],
            seed: int, 
            **kwargs
    ): 
        """
        :param n_estimators: count of estimators 
        :param base_estimator: base estimator (practically tree classifier) 
        :param seed: global seed 
        :param kwargs: keyword arguments of base estimator 
        """
        self.n_classes = None 
        self.error_history = []  # to track model learning process 
        self.n_estimators = n_estimators 
        self.rng = np.random.default_rng(seed) 
        self.base_estimator = base_estimator 
        self.base_estimator_kwargs = kwargs 
        # deduce which keywords are used to set seed for an estimator (sklearn or own tree implementation) 
        signature = inspect.signature(self.base_estimator.__init__) 
        self.seed_keyword = None 
        if 'seed' in signature.parameters: 
            self.seed_keyword = 'seed' 
        elif 'random_state' in signature.parameters: 
            self.seed_keyword = 'random_state' 
        self.estimators = []


    def create_new_estimator(
        self,
        seed: int
    ):
        """
        create new base estimator with proper keywords
        and new *unique* seed
        :param seed:
        :return: estimator
        """
        if self.seed_keyword is not None:
            self.base_estimator_kwargs[self.seed_keyword] = seed
        
        return self.base_estimator(**self.base_estimator_kwargs)


    @staticmethod 
    def get_estimator_error( 
            estimator: sklearn.base.BaseEstimator, 
            X: np.ndarray, 
            y: np.ndarray, 
            weights: np.ndarray 
    ): 
        """
        calculate weighted error of an estimator 
        :param estimator: 
        :param X:       [n_samples, n_features] 
        :param y:       [n_samples] 
        :param weights: [n_samples] 
        :return: 
        """
        estimator = estimator.fit(X, y, sample_weight=weights)
        predictions = estimator.predict(X)
        indicators = np.not_equal(predictions, y)
        return np.dot(weights, indicators) / sum(weights)


    def get_new_weights( 
        self, 
        true_labels: np.ndarray, 
        predictions: np.ndarray, 
        weights: np.ndarray 
    ): 
        """
        Calculate new weights according to SAMME.R scheme 
        :param true_labels: [n_samples] 
        :param predictions: [n_samples, n_classes] 
        :param weights:     [n_samples] 
        :return: normalized weights for next estimator fitting 
        """
        self.n_samples = len(true_labels)
        self.n_classes = np.shape(predictions)[1]

        y = np.full((self.n_samples, self.n_classes), -1 / (self.n_classes - 1))
        y[np.arange(self.n_samples), true_labels] = 1

        mults = np.array([np.dot(y[i], np.log(predictions[i] + self.eps))\
                               for i in range(self.n_samples)])
        mults *= (-(self.n_classes - 1) / self.n_classes)
        mults = np.exp(mults)

        new_weights = weights * mults
        new_weights /= np.sum(new_weights)
        return new_weights 


    def fit(self, X: np.ndarray, y: np.ndarray ):
        """ sequentially fit estimators with updated weights on each iteration
        :param X: [n_samples, n_features]
        :param y: [n_samples]
        :return: self
        """
        # sequentially fit each model and adjust weights for seed in
        # self.rng.choice(max(MyAdaBoostClassifier.big_number, self.n_estimators), size=self.n_estimators, replace=False):
        self.n_samples = np.shape(X)[0]
        weights = np.full(self.n_samples, 1 / self.n_samples)

        seeds = self.rng.choice(max(MyAdaBoostClassifier.big_number, self.n_estimators), size=self.n_estimators, replace=False)
        for m in range(self.n_estimators):
            estimator = self.create_new_estimator(seeds[m])
            estimator.fit(X, y, sample_weight=weights)
            self.estimators.append(estimator)
            self.error_history.append(self.get_estimator_error(estimator, X, y, weights))
            
            predictions = estimator.predict_proba(X)

            weights = self.get_new_weights(true_labels=y, predictions=predictions, weights=weights)
        return self
    

    def predict_proba( 
        self, 
        X: np.ndarray 
    ): 
        """
        predicts probability of each class 
        :param X: [n_samples, n_features] 
        :return: array of probabilities of a shape [n_samples, n_classes] 
        """
        self.n_test_samples = np.shape(X)[0]
        h = np.zeros(self.n_test_samples, self.n_classes)

        for m in range(self.n_estimators):
            logits = np.log(self.estimators[m].predict_proba(X) + self.eps)
            logits_sum = np.sum(logits, axis=1) / self.n_classes

            h_m = logits
            for i in range(self.n_classes):
                h_m[:, i] -= logits_sum
            h_m *= (self.n_classes - 1)
            h += h_m
        
        h /= (self.n_classes - 1)
        h = np.exp(h)
        for i in range(self.n_test_samples):
            h[i] /= np.sum(h[i])
        return h
 
    def predict( 
        self, 
        X: np.ndarray 
    ): 
        h = self.predict_proba(X)

        return np.argmax(h, axis=1) 
