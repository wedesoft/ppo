(ns ppo.ppo
    (:require
      [libpython-clj2.require :refer (require-python)]
      [libpython-clj2.python :refer (py.) :as py]
      [ppo.environment :refer (environment-observation environment-update environment-reward environment-done?
                               environment-truncate?)]))


(require-python '[torch :as torch])


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
            sample           (policy observation)
            action           (:action sample)
            logprob          (:logprob sample)
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


(defn probability-ratios
  "Probability ratios for a actions using updated policy and old policy"
  [{:keys [observations]} policy old-logprobs]
  (let [logprobs (:logprob (policy observations))]
    (torch/exp (py. (torch/sub logprobs old-logprobs) sum 1 :keepdim true))))


(defn clipped-surrogate-loss
  "Clipped surrogate loss (negative objective)"
  [probability-ratios advantages epsilon]
  (torch/mean
    (torch/neg
      (torch/min
        (torch/mul probability-ratios advantages)
        (torch/mul (torch/clamp probability-ratios (- 1.0 epsilon) (+ 1.0 epsilon)) advantages)))))
