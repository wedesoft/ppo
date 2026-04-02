(ns ppo.ppo
    (:require
      [ppo.environment :refer (environment-observation environment-update)]))


(defn sample-environment
  "Collect trajectory data from environment"
  [environment-factory policy n]
  (loop [environment (environment-factory) observations [] i n]
    (if (zero? i)
      {:observations observations}
      (let [observation (environment-observation environment)
            action      (policy observation)]
        (recur (environment-update environment action) (conj observations observation) (dec i))))))
