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
         logprobs          []
         next-observations []
         rewards           []
         dones             []
         truncates         []
         i                 n]
    (if (pos? i)
      (let [observation      (environment-observation state)
            action           (:action (policy observation))
            logprob          (:logprob (policy observation))
            reward           (environment-reward state action)
            done             (environment-done? state)
            truncate         (environment-truncate? state)
            next-state       (if (or done truncate) (environment-factory) (environment-update state action))
            next-observation (environment-observation next-state)]
        (recur next-state
               (conj observations observation)
               (conj actions action)
               (conj logprobs logprob)
               (conj next-observations next-observation)
               (conj rewards reward)
               (conj dones done)
               (conj truncates truncate)
               (dec i)))
      {:observations      observations
       :actions           actions
       :logprobs          logprobs
       :next-observations next-observations
       :rewards           rewards
       :dones             dones
       :truncates         truncates})))


(defn deltas
  "Compute difference between actual reward plus discounted estimate of next state and estimated value of current state"
  [{:keys [observations next-observations rewards dones]} critic gamma]
  (mapv (fn [observation next-observation reward done]
            (- (+ reward (if done 0.0 (* gamma (critic next-observation)))) (critic observation)))
        observations next-observations rewards dones))


(defn advantages
  "Compute advantages attributed to each action"
  [{:keys [dones]} deltas gamma lambda]
  (reverse
    (rest
      (reductions
        (fn [advantage [delta done]]
            (+ delta (if done 0.0 (* gamma lambda advantage))))
        0.0
        (reverse (map vector deltas dones))))))


(defn critic-target
  "Determine target values for critic"
  [{:keys [observations]} advantages critic]
  (map (fn [observation advantage] (+ (critic observation) advantage)) observations advantages))
