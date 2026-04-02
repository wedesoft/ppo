(ns ppo.t-ppo
    (:require
      [midje.sweet :refer :all]
      [ppo.environment :refer (Environment)]
      [ppo.ppo :refer :all]))


(defrecord TestEnvironment [state]
  Environment
  (environment-update [_this input] (->TestEnvironment (+ state (first input))))
  (environment-observation [_this] [(+ state 100)])
  (environment-done? [_this] (>= state 10))
  (environment-truncate? [_this] (< state 0))
  (environment-reward [_this] (- 5 (abs (- state 5)))))

(def counter (atom 0))

(defn test-env-factory []
  (let [counter (atom 0)]
    (fn [] (->TestEnvironment (swap! counter inc)))))


(facts "Generate samples from environment"
       (:observations (sample-environment (test-env-factory) 1)) => [[101]])
