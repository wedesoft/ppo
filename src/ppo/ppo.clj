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


(defn shuffle-samples
  "Random shuffle of samples"
  ([samples]
   (shuffle-samples samples (shuffle (range (count samples)))))
  ([samples indices]
   (zipmap (keys samples) (map #(map % indices) (vals samples)))))


(defn create-batches
  "Create mini batches from environment samples"
  [samples batch-size]
  (zipmap (keys samples) (map #(partition-all batch-size %) (vals samples))))


(defn deltas
  "Compute difference between actual reward plus discounted estimate of next state and estimated value of current state"
  [{:keys [observations next-observations rewards dones]} critic gamma]
  (mapv (fn [observation next-observation reward done]
            (- (+ reward (if done 0.0 (* gamma (critic next-observation)))) (critic observation)))
        observations next-observations rewards dones))


(defn advantages
  "Compute advantages attributed to each action"
  [{:keys [dones truncates]} deltas gamma lambda]
  (vec
    (reverse
    (rest
      (reductions
        (fn [advantage [delta done truncate]]
            (+ delta (if (or done truncate) 0.0 (* gamma lambda advantage))))
        0.0
        (reverse (map vector deltas dones truncates)))))))


(defn critic-target
  "Determine target values for critic"
  [{:keys [observations]} advantages critic]
  (map (fn [observation advantage] (+ (critic observation) advantage)) observations advantages))


(defn probability-ratios
  "Probability ratios for a actions using updated policy and old policy"
  [{:keys [observations logprobs actions]} logprob-of-action]
  (let [updated-logprobs (logprob-of-action observations actions)]
    (torch/exp (py. (torch/sub updated-logprobs logprobs) sum 1))))


(defn clipped-surrogate-loss
  "Clipped surrogate loss (negative objective)"
  [probability-ratios advantages epsilon]
  (torch/mean
    (torch/neg
      (torch/min
        (torch/mul probability-ratios advantages)
        (torch/mul (torch/clamp probability-ratios (- 1.0 epsilon) (+ 1.0 epsilon)) advantages)))))
