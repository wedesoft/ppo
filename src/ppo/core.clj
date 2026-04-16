(ns ppo.core
    (:gen-class)
    (:require [clojure.math :refer (PI cos sqrt)]
              [libpython-clj2.require :refer (require-python)]
              [libpython-clj2.python :refer (py.) :as py]
              [ppo.mlp :refer (Actor Critic adam-optimizer tolist tensor without-gradient entropy-of-distribution)]
              [ppo.ppo :refer (sample-with-advantage-and-critic-target actor-loss critic-loss)]
              [ppo.pendulum :refer (->Pendulum config setup action)]))

(require-python '[torch :as torch])


(defn pendulum-factory
  []
  (let [angle     (- (rand (* 2.0 PI)) PI)
        energy    (* (:gravitation config) (:length config) (+ 1.0 (cos angle)))
        max-speed (+ (sqrt (* 2.0 energy)) 2.0)
        velocity  (- (rand (* 2.0 max-speed)) max-speed)]
    (->Pendulum config (setup angle velocity))))


(defn -main [& _args]
  (let [factory          pendulum-factory
        actor            (Actor 3 100 1)
        critic           (Critic 3 100)
        n-epochs         100000
        n-updates        10
        gamma            0.98
        lambda           1.0
        epsilon          0.2
        batch-size       64
        n-batches        4
        checkpoint       100
        entropy-factor   (atom 0.001)
        entropy-decay    0.999
        lr               0.0002
        actor-optimizer  (adam-optimizer actor lr 0.0)
        critic-optimizer (adam-optimizer critic (* 3 lr) 0.0)]
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
                             (py. actor-optimizer step)
                             (swap! smooth-actor-loss (fn [x] (+ (* 0.95 x) (* 0.01 (tolist loss))))) ))
                    (doseq [batch samples]
                           (let [loss (critic-loss batch critic)]
                             (py. critic-optimizer zero_grad)
                             (py. loss backward)
                             (py. critic-optimizer step)
                             (swap! smooth-critic-loss (fn [x] (+ (* 0.97 x) (* 0.01 (tolist loss))))))))
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

(comment
  (require '[libpython-clj2.python :refer [py. py.. py.-] :as py])
  (require '[libpython-clj2.require :refer (require-python)])
  (require '[ppo.mlp :refer (Actor tensor)])
  (def actor (Actor 3 10 1))
  (py. actor load_state_dict (torch/load "actor.pt"))
)
