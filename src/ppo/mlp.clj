(ns ppo.mlp
    (:require [libpython-clj2.require :refer (require-python)]
              [libpython-clj2.python :refer (py.) :as py]))

(require-python '[builtins :as python]
                '[torch :as torch]
                '[torch.nn :as nn]
                '[torch.nn.functional :as F]
                '[torch.optim :as optim]
                '[torch.distributions :refer (Normal)])


(defmacro without-gradient
  "Execute body without gradient calculation"
  [& body]
  `(let [no-grad# (torch/no_grad)]
     (try
       (py. no-grad# ~'__enter__)
       ~@body
       (finally
         (py. no-grad# ~'__exit__ nil nil nil)))))


(defn tensor
  "Convert nested vector to tensor"
  [data]
  (torch/tensor data :dtype torch/float32))


(defn tolist
  "Convert tensor to nested vector"
  [tensor]
  (py/->jvm (py. tensor tolist)))


(defn toitem
  "Convert torch scalar value to float"
  [tensor]
  (py. tensor item))


(def Critic
  (py/create-class
    "Critic" [nn/Module]
    {"__init__"
     (py/make-instance-fn
       (fn [self observation-size hidden-units]
           (py. nn/Module __init__ self)
           (py/set-attrs!
             self
             {"fc1" (nn/Linear observation-size hidden-units)
              "fc2" (nn/Linear hidden-units hidden-units)
              "fc3" (nn/Linear hidden-units 1)})
           nil))
     "forward"
     (py/make-instance-fn
       (fn [self x]
           (let [x (py. self fc1 x)
                 x (torch/tanh x)
                 x (py. self fc2 x)
                 x (torch/tanh x)
                 x (py. self fc3 x)]
             (torch/squeeze x -1))))}))


(def Actor
  (py/create-class
    "Actor" [nn/Module]
    {"__init__"
     (py/make-instance-fn
       (fn [self observation-size hidden-units action-size]
           (py. nn/Module __init__ self)
           (py/set-attrs!
             self
             {"fc1" (nn/Linear observation-size hidden-units)
              "fc2" (nn/Linear hidden-units hidden-units)
              "fcmu" (nn/Linear hidden-units action-size)
              "fcsigma" (nn/Linear hidden-units action-size)})
           nil))
     "forward"
     (py/make-instance-fn
       (fn [self x]
           (let [x (py. self fc1 x)
                 x (torch/tanh x)
                 x (py. self fc2 x)
                 x (torch/tanh x)
                 mu (torch/tanh (py. self fcmu x))
                 sigma (F/softplus (py. self fcsigma x))]
             [mu sigma])))
     "deterministic_act"
     (py/make-instance-fn
       (fn [self x]
            (let [[mu _sigma] (py. self forward x)]
              mu)))
     "get_dist"
     (py/make-instance-fn
       (fn [self x]
           (let [[mu sigma] (py. self forward x)]
             (Normal mu sigma))))}))


(defn mse-loss
  "Mean square error cost function"
  []
  (nn/MSELoss))


(defn adam-optimizer
  "Adam optimizer"
  [model learning-rate weight-decay]
  (optim/Adam (py. model parameters) :lr learning-rate :weight_decay weight-decay))


(defn train
  "Train network for specified number of epochs"
  [optimizer model criterion batches epochs]
  (doseq [epoch (range epochs)]
         (doseq [[data label] batches]
                (py. optimizer zero_grad)
                (let [prediction (py. model __call__ data)
                      loss       (py. criterion __call__ prediction label)]
                  (py. loss backward)
                  (py. optimizer step)))))


(defn tensor-indeterministic-act
  "Sample action using actor network returning distribution"
  [actor]
  (fn tensor-indeterministic-act [observation]
      (let [dist    (py. actor get_dist observation)
            sample  (py. dist sample)
            action  (torch/clamp sample -1.0 1.0)
            logprob (py. dist log_prob action)]
        {:action action :logprob logprob})))


(defn indeterministic-act
  "Perform conversions to torch tensors, call tensor-indeterministic-act, and convert result back"
  [actor]
  (let [tensor-indeterministic-act (tensor-indeterministic-act actor)]
    (fn inteterministic-action [observation]
        (let [{:keys [action logprob]} (tensor-indeterministic-act (tensor observation))]
          {:action (tolist action) :logprob (tolist logprob)}))))
