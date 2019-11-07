# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from sklearn.linear_model import SGDClassifier
from sklearn.base import clone
import copy

class SGDModel:
    """Wrapper around `scikit-learn`'s SGD classifier allowing for external
    updating of model weights and a modified training interface.

    kwargs supplied to the constructor are passed to the underlying classifier.
    """
    
    def __init__(self, **kwargs):
        self.classifier = SGDClassifier(**kwargs)

    def get_clone(self, trained=False):
        """Create a clone of this classifier.

        trained: if `True`, maintains current state including trained model
            weights, interation counter, etc. Otherwise, returns an unfitted
            model with the same initialization params.
        """
        if trained:
            new_classifier = copy.deepcopy(self.classifier)
        else:
            new_classifier = clone(self.classifier)

        new_model = self.__class__()
        new_model.classifier = new_classifier
        return new_model

    def __repr__(self):
        return "SGDModel(\n{}\n)".format(self.classifier.__repr__())

    def set_weights(self, coef, intercept):
        """Update the current model weights."""
        self.classifier.coef_ = coef
        self.classifier.intercept_ = intercept

    def minibatch_update(self, X, y):
        """Run a single weight update on the given minibatch."""
        # TODO: implement. Need to consider how to set `t_` and `n_iter_`
        pass
