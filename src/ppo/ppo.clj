(ns ppo.ppo
    (:require
      [ppo.environment :refer (environment-observation)]))


(defn sample-environment
  "Collect trajectory data from environment"
  [environment-factory n]
  (let [environment (environment-factory)]
    {:observations [(environment-observation environment)]}))
