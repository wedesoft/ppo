(ns ppo.t-mlp
    (:require
      [midje.sweet :refer :all]
      [libpython-clj2.require :refer (require-python)]
      [libpython-clj2.python :refer (py.) :as py]
      [ppo.mlp :refer :all]))

(require-python '[torch :as torch])


(fact "Test critic network"
      (let [zero-critic (Critic 2 5)]
        (without-gradient
          (doseq [param (py. zero-critic parameters)]
                 (py. param zero_)))
        (py. zero-critic eval)
        (tolist (without-gradient (py. zero-critic __call__ (tensor [[0 0] [0 0] [0 0]])))) => [[0.0] [0.0] [0.0]]))


(fact "Mean square error cost function"
      (let [criterion (mse-loss)]
        (without-gradient
          (toitem (criterion (tensor [[0.0] [0.0] [0.0]]) (tensor [[0.0] [0.0] [0.0]]))) => 0.0
          (toitem (criterion (tensor [[0.0] [0.0] [0.0]]) (tensor [[1.0] [1.0] [1.0]]))) => 1.0
          (toitem (criterion (tensor [[-1.0] [-1.0] [-1.0]]) (tensor [[1.0] [1.0] [1.0]])))  => 4.0)))


(fact "Train network"
      (let [model     (Critic 1 2)
            optimizer (adam-optimizer model 0.1 0.001)
            batches   [[(tensor [[0.0] [0.0]]) (tensor [[0.0] [0.0]])] [(tensor [[1.0] [1.0]]) (tensor [[1.0] [1.0]])]]
            criterion (mse-loss)]
        (py. model train)
        (train optimizer model criterion batches 100)
        (py. model eval)
        (without-gradient
          (toitem (criterion (py. model __call__ (tensor [[0.0] [1.0]])) (tensor [[0.0] [1.0]]))) => (roughly 0.0 1e-3))))
