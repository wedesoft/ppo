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
  (environment-reward [_this _action] (- (abs (- state 5)))))


(defn test-env-factory [] (constantly (->TestEnvironment 1)))
(defn stop-at-102 [observation] (if (>= (first observation) 102) [0] [1]))
(defn feedback-state [observation] [(- (first observation) 100)])

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
       (:next-observations (sample-environment (test-env-factory) (constantly [3]) 5)) => [[104] [107] [110] [101] [104]]
       (:actions (sample-environment (test-env-factory) feedback-state 3)) => [[1] [2] [4]])


(defn linear-critic [observation] (first observation))

(facts "Compute difference between actual reward plus discounted estimate of next state and estimated value of current state"
       (deltas {:observations [[4]] :next-observations [[3]] :rewards [0] :dones [false]} (constantly 0) 1.0) => [0.0]
       (deltas {:observations [[4]] :next-observations [[3]] :rewards [1] :dones [false]} (constantly 0) 1.0) => [1.0]
       (deltas {:observations [[4]] :next-observations [[3]] :rewards [1] :dones [false]} linear-critic 1.0) => [0.0]
       (deltas {:observations [[2]] :next-observations [[1]] :rewards [1] :dones [false]} linear-critic 0.5) => [-0.5]
       (deltas {:observations [[4] [3]] :next-observations [[3] [2]] :rewards [2 3] :dones [false false]} linear-critic 1.0)
       => [1.0 2.0]
       (deltas {:observations [[4]] :next-observations [[3]] :rewards [4] :dones [true]} linear-critic 1.0) => [0.0])


(facts "Compute advantages attributed to each action"
       (advantages {:dones [false]} [0.0] 1.0 1.0) => [0.0]
       (advantages {:dones [false]} [1.0] 1.0 1.0) => [1.0]
       (advantages {:dones [false false]} [2.0 3.0] 1.0 1.0) => [5.0 3.0]
       (advantages {:dones [false false]} [2.0 3.0] 0.5 1.0) => [3.5 3.0]
       (advantages {:dones [false false]} [2.0 3.0] 1.0 0.5) => [3.5 3.0]
       (advantages {:dones [true false]} [2.0 3.0] 1.0 1.0) => [2.0 3.0])
