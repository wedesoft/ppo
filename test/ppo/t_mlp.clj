(ns ppo.t-mlp
    (:require
      [midje.sweet :refer :all]
      [libpython-clj2.require :refer (require-python)]
      [libpython-clj2.python :refer (py.) :as py]
      [ppo.mlp :refer :all]))

(require-python '[torch :as torch])


(def critic (Critic 2 5))
(without-gradient
  (doseq [param (py. critic parameters)]
         (py. param zero_)))

(fact "Test critic network"
      (let [critic (Critic 2 5)]
        (without-gradient
          (doseq [param (py. critic parameters)]
                 (py. param zero_)))
        (tolist (without-gradient (py. critic __call__ (tensor [[0 0] [0 0] [0 0]]))))) => [[0.0] [0.0] [0.0]])


(fact "Mean square error cost function"
      (let [criterion (mse-loss)]
        (without-gradient
          (toitem (criterion (tensor [[0.0] [0.0] [0.0]]) (tensor [[0.0] [0.0] [0.0]]))) => 0.0
          (toitem (criterion (tensor [[0.0] [0.0] [0.0]]) (tensor [[1.0] [1.0] [1.0]]))) => 1.0
          (toitem (criterion (tensor [[-1.0] [-1.0] [-1.0]]) (tensor [[1.0] [1.0] [1.0]])))  => 4.0)))
