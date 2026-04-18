(ns ppo.core
    (:gen-class)
    (:require [clojure.math :refer (PI cos sqrt)]
              [libpython-clj2.require :refer (require-python)]
              [libpython-clj2.python :refer (py.) :as py]
              [ppo.mlp :refer (Actor Critic adam-optimizer tolist tensor without-gradient entropy-of-distribution)]
              [ppo.ppo :refer (sample-with-advantage-and-critic-target actor-loss critic-loss)]
              [ppo.pendulum :refer (->Pendulum config setup action)]))

(require-python '[torch :as torch]
                '[torch.nn.utils :as utils])


(defn pendulum-factory
  []
  (let [angle     (- (rand (* 2.0 PI)) PI)
        max-speed (:max-speed config)
        velocity  (- (rand (* 2.0 max-speed)) max-speed)]
    (->Pendulum config (setup angle velocity))))


(defn -main [& _args]
  (let [factory          pendulum-factory
        actor            (Actor 3 64 1)
        critic           (Critic 3 64)
        n-epochs         100000
        n-updates        10
        gamma            0.99
        lambda           1.0
        epsilon          0.2
        n-batches        8
        batch-size       50
        checkpoint       100
        entropy-factor   (atom 0.1)
        entropy-decay    0.999
        lr               5e-5
        weight-decay     1e-4
        actor-optimizer  (adam-optimizer actor lr weight-decay)
        critic-optimizer (adam-optimizer critic lr weight-decay)]
    (when (.exists (java.io.File. "actor.pt"))
      (py. actor load_state_dict (torch/load "actor.pt")))
    (when (.exists (java.io.File. "critic.pt"))
      (py. critic load_state_dict (torch/load "critic.pt")))
    (doseq [epoch (range n-epochs)]
           (let [smooth-actor-loss  (atom 0.0)
                 smooth-critic-loss (atom 0.0)
                 samples         (sample-with-advantage-and-critic-target factory actor critic (* batch-size n-batches)
                                                                          batch-size gamma lambda)]
             (doseq [k (range n-updates)]
                    (doseq [batch samples]
                           (let [loss (actor-loss batch actor epsilon @entropy-factor)]
                             (py. actor-optimizer zero_grad)
                             (py. loss backward)
                             (utils/clip_grad_norm_(py. actor parameters) 0.5)
                             (py. actor-optimizer step)
                             (swap! smooth-actor-loss (fn [x] (+ (* 0.95 x) (* 0.05 (tolist loss))))) ))
                    (doseq [batch samples]
                           (let [loss (critic-loss batch critic)]
                             (py. critic-optimizer zero_grad)
                             (py. loss backward)
                             (py. critic-optimizer step)
                             (swap! smooth-critic-loss (fn [x] (+ (* 0.95 x) (* 0.05 (tolist loss))))))))
             (println "Epoch:" epoch
                      "Actor Loss:" @smooth-actor-loss
                      "Critic Loss:" @smooth-critic-loss
                      "Entropy Factor:" @entropy-factor))
           (without-gradient
             (doseq [input [[1 0 -1.0] [1 0 1.0] [0 -1 -1.0] [0 -1 1.0] [0 1 -1.0] [0 1 1.0] [-1 0 -1.0] [-1 0 1.0]]]
                    (println input
                             "->" (action (tolist (py. actor deterministic_act (tensor input))))
                             "entropy" (tolist (entropy-of-distribution actor (tensor input))))))
           (swap! entropy-factor * entropy-decay)
           (when (= (mod epoch checkpoint) (dec checkpoint))
             (println "Saving models")
             (torch/save (py. actor state_dict) "actor.pt")
             (torch/save (py. critic state_dict) "critic.pt")))
    (torch/save (py. actor state_dict) "actor.pt")
    (torch/save (py. critic state_dict) "critic.pt")
    (System/exit 0)))
