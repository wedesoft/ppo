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

(facts "Test critic network"
       (py/->jvm (py. (without-gradient (py. critic __call__ (tensor [[0 0] [0 0] [0 0]]))) tolist))
       => [[0.0] [0.0] [0.0]])
