(ns ppo.mlp
    (:require [libpython-clj2.require :refer (require-python)]
              [libpython-clj2.python :refer (py.) :as py]))

(require-python '[torch :as torch]
                '[torch.nn :as nn]
                '[torch.optim :as optim])


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
             x)))}))


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
