(ns ppo.t-ppo
    (:require
      [midje.sweet :refer :all]
      [ppo.environment :refer (Environment)]
      [ppo.mlp :refer (tensor tolist Actor tensor-indeterministic-act indeterministic-act)]
      [ppo.ppo :refer :all]))


(defrecord TestEnvironment [state]
  Environment
  (environment-update [_this action] (->TestEnvironment (+ state (first action))))
  (environment-observation [_this] [(+ state 100)])
  (environment-done? [_this] (>= state 10))
  (environment-truncate? [_this] (< state 0))
  (environment-reward [_this _action] (- (abs (- state 5)))))


(defn test-env-factory [] (constantly (->TestEnvironment 1)))
(defn constant-value [a] (fn [observation] {:action [a] :logprob [0]}))
(defn stop-at-102 [observation] {:action (if (>= (first observation) 102) [0] [1]) :logprob [0]})
(defn feedback-state [observation] {:action [(- (first observation) 100)] :logprob [(- 100 (first observation))]})

(facts "Generate samples from environment"
       (:observations (sample-environment (test-env-factory) (constant-value 0) 1)) => [[101]]
       (:observations (sample-environment (test-env-factory) (constant-value 0) 2)) => [[101] [101]]
       (:observations (sample-environment (test-env-factory) (constant-value 1) 2)) => [[101] [102]]
       (:observations (sample-environment (test-env-factory) stop-at-102 3)) => [[101] [102] [102]]
       (:rewards (sample-environment (test-env-factory) (constant-value 1) 5)) => [-4 -3 -2 -1 0]
       (:dones (sample-environment (test-env-factory) (constant-value 3) 4)) => [false false false true]
       (:observations (sample-environment (test-env-factory) (constant-value 3) 5)) => [[101] [104] [107] [110] [101]]
       (:truncates (sample-environment (test-env-factory) (constant-value -1) 3)) => [false false true]
       (:observations (sample-environment (test-env-factory) (constant-value -1) 4)) => [[101] [100] [99] [101]]
       (:next-observations (sample-environment (test-env-factory) (constant-value 3) 5)) => [[104] [107] [110] [101] [104]]
       (:actions (sample-environment (test-env-factory) feedback-state 3)) => [[1] [2] [4]]
       (:logprobs (sample-environment (test-env-factory) (constant-value 0) 1))  => [[0]]
       (:logprobs (sample-environment (test-env-factory) feedback-state 3)) => [[-1] [-2] [-4]])


(fact "Integration test sampling environment"
      (let [factory (test-env-factory)
            actor   (Actor 1 5 1)
            samples (sample-environment (test-env-factory) (indeterministic-act actor) 8)]
        (count (:observations samples)) => 8))


(facts "Random shuffle of samples"
       (let [samples  {:observations [[101] [102] [103] [104]]
                       :actions [[1] [2] [3] [4]]
                       :logprobs [[-1] [-2] [-3] [-4]]
                       :next-observations [[102] [103] [104] [105]]
                       :rewards [-4 -3 -2 -1]
                       :dones [false false false true]
                       :truncates [false true false false]}
             shuffled (shuffle-samples samples [0 3 2 1])]
         (:observations shuffled) => [[101] [104] [103] [102]]
         (:actions shuffled) => [[1] [4] [3] [2]]
         (:logprobs shuffled) => [[-1] [-4] [-3] [-2]]
         (:next-observations shuffled) => [[102] [105] [104] [103]]
         (:rewards shuffled) => [-4 -1 -2 -3]
         (:dones shuffled) => [false true false false]
         (:truncates shuffled) => [false false false true]))


(facts "Create batches from samples"
       (let [samples {:observations [[101] [102] [103] [104]]
                      :actions [[1] [2] [3] [4]]
                      :logprobs [[-1] [-2] [-3] [-4]]
                      :next-observations [[102] [103] [104] [105]]
                      :rewards [-4 -3 -2 -1]
                      :dones [false false false true]
                      :truncates [false true false false]}
             batches (create-batches samples 2)]
         (first (:observations batches)) => [[101] [102]]
         (second (:observations batches)) => [[103] [104]]
         (first (:actions batches)) => [[1] [2]]
         (second (:actions batches)) => [[3] [4]]
         (first (:logprobs batches)) => [[-1] [-2]]
         (second (:logprobs batches)) => [[-3] [-4]]
         (first (:next-observations batches)) => [[102] [103]]
         (second (:next-observations batches)) => [[104] [105]]
         (first (:rewards batches)) => [-4 -3]
         (second (:rewards batches)) => [-2 -1]
         (first (:dones batches)) => [false false]
         (second (:dones batches)) => [false true]
         (first (:truncates batches)) => [false true]
         (second (:truncates batches)) => [false false]))


(defn linear-critic [observation] (first observation))

(facts "Compute difference between actual reward plus discounted estimate of next state and estimated value of current state"
       (deltas {:observations [[4]] :next-observations [[3]] :rewards [0] :dones [false]} (constantly 0) 1.0) => [0.0]
       (deltas {:observations [[4]] :next-observations [[3]] :rewards [1] :dones [false]} (constantly 0) 1.0) => [1.0]
       (deltas {:observations [[4]] :next-observations [[3]] :rewards [1] :dones [false]} linear-critic 1.0) => [0.0]
       (deltas {:observations [[2]] :next-observations [[1]] :rewards [1] :dones [false]} linear-critic 0.5) => [-0.5]
       (deltas {:observations [[4] [3]] :next-observations [[3] [2]] :rewards [2 3] :dones [false false]} linear-critic 1.0)
       => [1.0 2.0]
       (deltas {:observations [[4]] :next-observations [[3]] :rewards [4] :dones [true]} linear-critic 1.0) => [0.0]
       (deltas {:observations [[4]] :next-observations [[3]] :rewards [4] :dones [true]} linear-critic 1.0) => [0.0])


(facts "Compute advantages attributed to each action"
       (advantages {:dones [false] :truncates [false]} [0.0] 1.0 1.0) => [0.0]
       (advantages {:dones [false] :truncates [false]} [1.0] 1.0 1.0) => [1.0]
       (advantages {:dones [false false] :truncates [false false]} [2.0 3.0] 1.0 1.0) => [5.0 3.0]
       (advantages {:dones [false false] :truncates [false false]} [2.0 3.0] 0.5 1.0) => [3.5 3.0]
       (advantages {:dones [false false] :truncates [false false]} [2.0 3.0] 1.0 0.5) => [3.5 3.0]
       (advantages {:dones [true false] :truncates [false false]} [2.0 3.0] 1.0 1.0) => [2.0 3.0]
       (advantages {:dones [false false] :truncates [true false]} [2.0 3.0] 1.0 1.0) => [2.0 3.0])


(facts "Target values for critic"
       (critic-target {:observations [[4]]} [0] (constantly 0)) => [0]
       (critic-target {:observations [[4]]} [2] (constantly 0)) => [2]
       (critic-target {:observations [[4]]} [0] linear-critic) => [4]
       (critic-target {:observations [[4]]} [3] linear-critic) => [7])


(defn action-prob [p] (fn [observations] {:action [[0]] :logprob (tensor [[p]])}))
(defn identity-prob [observations] {:action [[0]] :logprob observations})

(facts "Probability ratios for a actions using updated policy and old policy"
       (tolist (probability-ratios {:observations (tensor [[4]]) :logprobs (tensor [[0]])} (action-prob 0))) => [[1.0]]
       (tolist (probability-ratios {:observations (tensor [[4]]) :logprobs (tensor [[1]])} (action-prob 0))) => [[0.3678794503211975]]
       (tolist (probability-ratios {:observations (tensor [[4]]) :logprobs (tensor [[0]])} (action-prob 1))) => [[2.7182817459106445]]
       (tolist (probability-ratios {:observations (tensor [[1]]) :logprobs (tensor [[0]])} identity-prob)) => [[2.7182817459106445]]
       (tolist (probability-ratios {:observations (tensor [[2 3]]) :logprobs (tensor [[2 3]])} identity-prob)) => [[1.0]]
       (tolist (probability-ratios {:observations (tensor [[0 1]]) :logprobs (tensor [[0 0]])} identity-prob)) => [[2.7182817459106445]]
       (tolist (probability-ratios {:observations (tensor [[0 0]]) :logprobs (tensor [[0 1]])} identity-prob)) => [[0.3678794503211975]])


(facts "Clipped surrogate loss (negative objective)"
       ;; zero advantage
       (tolist (clipped-surrogate-loss (tensor [[1.0]]) (tensor [[0.0]]) 0.25)) => 0.0
       ;; positive advantage
       (tolist (clipped-surrogate-loss (tensor [[1.0]]) (tensor [[3.0]]) 0.25)) => -3.0
       (tolist (clipped-surrogate-loss (tensor [[1.25]]) (tensor [[3.0]]) 0.25)) => -3.75
       (tolist (clipped-surrogate-loss (tensor [[2.0]]) (tensor [[3.0]]) 0.25)) => -3.75
       (tolist (clipped-surrogate-loss (tensor [[0.0]]) (tensor [[3.0]]) 0.25)) => 0.0
       ;; negative advantage
       (tolist (clipped-surrogate-loss (tensor [[1.0]]) (tensor [[-3.0]]) 0.25)) => 3.0
       (tolist (clipped-surrogate-loss (tensor [[0.75]]) (tensor [[-3.0]]) 0.25)) => 2.25
       (tolist (clipped-surrogate-loss (tensor [[0.0]]) (tensor [[-3.0]]) 0.25)) => 2.25
       (tolist (clipped-surrogate-loss (tensor [[2.0]]) (tensor [[-3.0]]) 0.25)) => 6.0)
