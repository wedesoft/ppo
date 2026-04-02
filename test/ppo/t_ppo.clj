(ns ppo.t-ppo
    (:require
      [midje.sweet :refer :all]
      [ppo.environment :refer (Environment)]
      [ppo.ppo :refer :all]))


(defrecord TestEnvironment [state]
  Environment
  (environment-update [_this action] (->TestEnvironment (+ state (first action))))
  (environment-observation [_this] [(+ state 100)])
  (environment-done? [_this] (>= state 10))
  (environment-truncate? [_this] (< state 0))
  (environment-reward [_this] (- 5 (abs (- state 5)))))

(def counter (atom 0))


(defn test-env-factory [] (let [counter (atom 0)] (fn [] (->TestEnvironment (swap! counter inc)))))
(defn stop-at-102 [observation] (if (>= (first observation) 102) [0] [1]))

(facts "Generate samples from environment"
       (:observations (sample-environment (test-env-factory) (constantly [0]) 1)) => [[101]]
       (:observations (sample-environment (test-env-factory) (constantly [0]) 2)) => [[101] [101]]
       (:observations (sample-environment (test-env-factory) (constantly [1]) 2)) => [[101] [102]]
       (:observations (sample-environment (test-env-factory) stop-at-102 3)) => [[101] [102] [102]])
