(ns ppo.t-mlp
    (:require
      [midje.sweet :refer :all]
      [libpython-clj2.require :refer (require-python)]
      [libpython-clj2.python :refer (py.) :as py]
      [ppo.mlp :refer :all]))

(require-python '[torch :as torch])


(fact "Test critic network"
      (without-gradient
        (let [zero-critic (Critic 2 5)]
          (doseq [param (py. zero-critic parameters)]
                 (py. param zero_))
          (py. zero-critic eval)
          (tolist (zero-critic (tensor [[0 0] [0 0] [0 0]]))) => [0.0 0.0 0.0])))


(fact "Mean square error cost function"
      (let [criterion (mse-loss)]
        (without-gradient
          (toitem (criterion (tensor [[0.0] [0.0] [0.0]]) (tensor [[0.0] [0.0] [0.0]]))) => 0.0
          (toitem (criterion (tensor [[0.0] [0.0] [0.0]]) (tensor [[1.0] [1.0] [1.0]]))) => 1.0
          (toitem (criterion (tensor [[-1.0] [-1.0] [-1.0]]) (tensor [[1.0] [1.0] [1.0]])))  => 4.0)))


(fact "Train network"
      (let [model     (Critic 1 2)
            optimizer (adam-optimizer model 0.1 0.001)
            batches   [[(tensor [[0.0] [0.0]]) (tensor [0.0 0.0])] [(tensor [[1.0] [1.0]]) (tensor [1.0 1.0])]]
            criterion (mse-loss)]
        (py. model train)
        (train optimizer model criterion batches 100)
        (py. model eval)
        (without-gradient
          (toitem (criterion (model (tensor [[0.0] [1.0]])) (tensor [0.0 1.0]))) => (roughly 0.0 1e-3))))


(facts "Test actor network"
       (without-gradient
         (let [zero-actor (Actor 2 5 1)]
           (doseq [param (py. zero-actor parameters)]
                  (py. param zero_))
           (py. zero-actor eval)
           (let [result (zero-actor (tensor [[0 0] [0 0] [0 0]]))]
             (tolist (first result)) => [[0.0] [0.0] [0.0]]
             (tolist (second result)) => [[0.6931471824645996] [0.6931471824645996] [0.6931471824645996]])
           (tolist (py. zero-actor deterministic_act (tensor [[0 0]]))) => [[0.0]]
           (tolist (py/py.- (py. zero-actor get_dist (tensor [[0 0]])) mean)) => [[0.0]]
           (tolist (py/py.- (py. zero-actor get_dist (tensor [[0 0]])) stddev)) => [[0.6931471824645996]]
           (:action (tolist ((tensor-indeterministic-act zero-actor) (tensor [[0 0]])))) => some?
           (:logprob (tolist ((tensor-indeterministic-act zero-actor) (tensor [[0 0]])))) => some?
           (:action ((indeterministic-act zero-actor) [[0 0]])) => some?
           (:logprob ((indeterministic-act zero-actor) [[0 0]])) => some?)))
