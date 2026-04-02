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
  (environment-reward [_this] (- (abs (- state 5)))))

(def counter (atom 0))


(defn test-env-factory [] (constantly (->TestEnvironment 1)))
(defn stop-at-102 [observation] (if (>= (first observation) 102) [0] [1]))

(facts "Generate samples from environment"
       (:observations (sample-environment (test-env-factory) (constantly [0]) 1)) => [[101]]
       (:observations (sample-environment (test-env-factory) (constantly [0]) 2)) => [[101] [101]]
       (:observations (sample-environment (test-env-factory) (constantly [1]) 2)) => [[101] [102]]
       (:observations (sample-environment (test-env-factory) stop-at-102 3)) => [[101] [102] [102]]
       (:rewards (sample-environment (test-env-factory) (constantly [1]) 5)) => [-4 -3 -2 -1 0]
       (:dones (sample-environment (test-env-factory) (constantly [3]) 4)) => [false false false true]
       (:observations (sample-environment (test-env-factory) (constantly [3]) 5)) => [[101] [104] [107] [110] [101]]
       (:truncates (sample-environment (test-env-factory) (constantly [-1]) 3)) => [false false true]
       (:observations (sample-environment (test-env-factory) (constantly [-1]) 4)) => [[101] [100] [99] [101]]
       (:next-observations (sample-environment (test-env-factory) (constantly [3]) 5)) => [[104] [107] [110] [101] [104]])
