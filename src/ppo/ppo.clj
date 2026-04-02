(ns ppo.ppo
    (:require
      [ppo.environment :refer (environment-observation environment-update environment-reward environment-done?
                               environment-truncate?)]))


(defn sample-environment
  "Collect trajectory data from environment"
  [environment-factory policy n]
  (loop [state             (environment-factory)
         observations      []
         actions           []
         next-observations []
         rewards           []
         dones             []
         truncates         []
         i                 n]
    (if (zero? i)
      {:observations      observations
       :actions           actions
       :next-observations next-observations
       :rewards           rewards
       :dones             dones
       :truncates         truncates}
      (let [observation      (environment-observation state)
            action           (policy observation)
            reward           (environment-reward state)
            done             (environment-done? state)
            truncate         (environment-truncate? state)
            next-state       (if (or done truncate) (environment-factory) (environment-update state action))
            next-observation (environment-observation next-state)]
        (recur next-state
               (conj observations observation)
               (conj actions action)
               (conj next-observations next-observation)
               (conj rewards reward)
               (conj dones done)
               (conj truncates truncate)
               (dec i))))))
