(ns ppo.ppo
    (:require
      [ppo.environment :refer (environment-observation environment-update environment-reward environment-done?
                               environment-truncate?)]))


(defn done?
  "Decide whether a run is finished or aborted"
  [environment]
  (or (environment-done? environment) (environment-truncate? environment)))


(defn sample-environment
  "Collect trajectory data from environment"
  [environment-factory policy n]
  (loop [state             (environment-factory)
         observations      []
         next-observations []
         rewards           []
         dones             []
         i                 n]
    (if (zero? i)
      {:observations      observations
       :next-observations next-observations
       :rewards           rewards
       :dones             dones}
      (let [observation      (environment-observation state)
            reward           (environment-reward state)
            done             (done? state)
            action           (policy observation)
            next-state       (if done (environment-factory) (environment-update state action))
            next-observation (environment-observation next-state)]
        (recur next-state
               (conj observations observation)
               (conj next-observations next-observation)
               (conj rewards reward)
               (conj dones done)
               (dec i))))))
