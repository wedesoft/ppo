(ns ppo.core
    (:gen-class)
    (:require [clojure.math :refer (PI)]
              [libpython-clj2.require :refer (require-python)]
              [libpython-clj2.python :refer (py.) :as py]
              [ppo.mlp :refer (Actor Critic adam-optimizer tolist)]
              [ppo.ppo :refer (sample-with-advantage-and-critic-target actor-loss critic-loss)]
              [ppo.pendulum :refer (->Pendulum config setup)]))

(require-python '[torch :as torch])


(defn pendulum-factory
  []
  (->Pendulum config (setup (rand (* 2 PI)) (- (rand 1.0) 0.5))))


(defn -main [& _args]
  (let [factory          pendulum-factory
        actor            (Actor 3 10 1)
        critic           (Critic 3 10)
        n-epochs         20
        gamma            0.8
        lambda           1.0
        epsilon          0.2
        batch-size       64
        n-batches        16
        checkpoint       10
        actor-optimizer  (adam-optimizer actor 0.01 0.001)
        critic-optimizer (adam-optimizer critic 0.01 0.001)]
    ; (when (.exists (java.io.File. "actor.pt"))
    ;   (py. actor load_state_dict (torch/load "actor.pt")))
    ; (when (.exists (java.io.File. "critic.pt"))
    ;   (py. critic load_state_dict (torch/load "critic.pt")))
    (doseq [epoch (range n-epochs)]
           (let [smooth-actor-loss  (atom 0.0)
                 smooth-critic-loss (atom 0.0)
                 samples         (sample-with-advantage-and-critic-target factory actor critic (* batch-size n-batches)
                                                                          batch-size gamma lambda)]
             (doseq [batch samples]
                    (py. actor-optimizer zero_grad)
                    (let [loss (actor-loss batch actor epsilon)]
                      (py. loss backward)
                      (py. actor-optimizer step)
                      (swap! smooth-actor-loss (fn [x] (+ (* 0.95 x) (* 0.01 (tolist loss))))) ))
             (doseq [batch samples]
                    (py. critic-optimizer zero_grad)
                    (let [loss (critic-loss batch critic)]
                      (py. loss backward)
                      (py. critic-optimizer step)
                      (swap! smooth-critic-loss (fn [x] (+ (* 0.97 x) (* 0.01 (tolist loss)))))))
             (println "Epoch:" epoch
                      "Actor Loss:" @smooth-actor-loss
                      "Critic Loss:" @smooth-critic-loss))
           (when (= (mod epoch checkpoint) (dec checkpoint))
             (torch/save (py. actor state_dict) "actor.pt")
             (torch/save (py. critic state_dict) "critic.pt")))
    (torch/save (py. actor state_dict) "actor.pt")
    (torch/save (py. critic state_dict) "critic.pt")
    (System/exit 0)))

(comment
  (require '[libpython-clj2.python :refer [py. py.. py.-] :as py])
  (require '[libpython-clj2.require :refer (require-python)])
  (require '[ppo.mlp :refer (Actor tensor)])
  (def actor (Actor 3 10 1))
  (py. actor load_state_dict (torch/load "actor.pt"))
)
