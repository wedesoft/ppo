(ns ppo.core
    (:gen-class)
    (:require [clojure.math :refer (PI)]
              [libpython-clj2.python :refer (py.) :as py]
              [ppo.mlp :refer (Actor Critic adam-optimizer tolist)]
              [ppo.ppo :refer (sample-with-advantage-and-critic-target actor-loss critic-loss)]
              [ppo.pendulum :refer (->Pendulum config setup)]))


(defn pendulum-factory
  []
  (->Pendulum config (setup (rand (* 2 PI)))))


(defn -main [& _args]
  (let [factory          pendulum-factory
        actor            (Actor 3 10 1)
        critic           (Critic 3 10)
        n                100
        c                50
        actor-optimizer  (adam-optimizer actor 0.1 0.001)
        critic-optimizer (adam-optimizer critic 0.1 0.001)]
    (doseq [k (range n)]
           (let [sum-actor-loss  (atom 0.0)
                 sum-critic-loss (atom 0.0)
                 samples         (sample-with-advantage-and-critic-target factory actor critic 128 16 0.9 1.0)]
             (doseq [epoch (range c)]
                    (doseq [batch samples]
                           (py. actor-optimizer zero_grad)
                           (let [loss (actor-loss batch actor 0.2)]
                             (py. loss backward)
                             (py. actor-optimizer step)
                             (swap! sum-actor-loss + (tolist loss)))))
             (doseq [epoch (range c)]
                    (doseq [batch samples]
                           (py. critic-optimizer zero_grad)
                           (let [loss (critic-loss batch critic)]
                             (py. loss backward)
                             (py. critic-optimizer step)
                             (swap! sum-critic-loss + (tolist loss)))))
             (println "Epoch: " k " Actor Loss: " (/ @sum-actor-loss c) " Critic Loss: " (/ @sum-critic-loss c))))
    (System/exit 0)))
